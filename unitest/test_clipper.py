import unittest
import asyncio
import copy
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Set, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the class to test
from simplex.context import RollContextClipper, identify_openai_function_calling
from simplex.basics import ModelResponse, ModelInput, UserNotify, AgentLoopStateEdit
from simplex.io.base import UserOutputInterface


class TestIdentifyOpenAIFunctionCalling(unittest.TestCase):
    """Test the identify_openai_function_calling function"""
    
    def test_no_function_calls(self):
        """Test with messages that have no function calls"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        count, indices = identify_openai_function_calling(messages)
        self.assertEqual(count, 0)
        self.assertEqual(indices, set())
    
    def test_with_tool_calls(self):
        """Test with assistant tool calls"""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "London"}'}
                    }
                ]
            },
            {"role": "user", "content": "Thanks!"}
        ]
        
        count, indices = identify_openai_function_calling(messages)
        self.assertEqual(count, 1)
        self.assertEqual(indices, {1})
    
    def test_with_tool_responses(self):
        """Test with tool responses"""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather"}}]
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72°F"},
            {"role": "assistant", "content": "The weather is sunny and 72°F."}
        ]
        
        count, indices = identify_openai_function_calling(messages)
        # Should count both tool call and tool response
        self.assertEqual(count, 2)
        self.assertEqual(indices, {1, 2})
    
    def test_multiple_tool_calls(self):
        """Test with multiple tool calls in one message"""
        messages = [
            {"role": "user", "content": "Get weather for two cities"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "London"}'}},
                    {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
                ]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Rainy, 60°F"},
            {"role": "tool", "tool_call_id": "call_2", "content": "Cloudy, 65°F"}
        ]
        
        count, indices = identify_openai_function_calling(messages)
        self.assertEqual(count, 3)  # 1 tool call message + 2 tool responses
        self.assertEqual(indices, {1, 2, 3})


class TestRollContextClipper(unittest.TestCase):
    """Test the RollContextClipper class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.clipper = RollContextClipper(
            instance_id="test_instance",
            max_context_tokens=1000,
            threshold_ratio=0.8,
            keep_fc_msgs=3
        )
        self.mock_output = Mock(spec=UserOutputInterface)
        self.mock_output.push_message = AsyncMock()
        
    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.clipper.max_context_tokens, 1000)
        self.assertEqual(self.clipper.threshold_ratio, 0.8)
        self.assertEqual(self.clipper.keep_fc_msgs, 3)
        self.assertEqual(self.clipper.current_max_tokens, 0)
        self.assertEqual(self.clipper.total_clipped_msgs, 0)
        self.assertIsNone(self.clipper.output_interface)
        
    def test_initialization_with_custom_identify_function(self):
        """Test initialization with custom identify function"""
        def custom_identify(messages):
            return 1, {0}
        
        clipper = RollContextClipper(
            max_context_tokens=2000,
            identify_function=custom_identify
        )
        self.assertEqual(clipper.identify_function, custom_identify)
    
    @pytest.mark.asyncio
    async def test_bind_io(self):
        """Test binding output interface"""
        await self.clipper.bind_io(self.mock_output)
        self.assertEqual(self.clipper.output_interface, self.mock_output)
    
    @pytest.mark.asyncio
    async def test_start_loop_async(self):
        """Test start_loop_async resets metrics"""
        self.clipper.current_max_tokens = 500
        self.clipper.total_clipped_msgs = 10
        
        await self.clipper.start_loop_async()
        
        self.assertEqual(self.clipper.current_max_tokens, 0)
        self.assertEqual(self.clipper.total_clipped_msgs, 0)
    
    @pytest.mark.asyncio
    async def test_after_response_async_updates_max_tokens(self):
        """Test after_response_async updates max tokens correctly"""
        self.clipper.current_max_tokens = 100
        
        response = ModelResponse(token_cost=150)
        await self.clipper.after_response_async(response)
        
        self.assertEqual(self.clipper.current_max_tokens, 150)
        
        # Test with lower token cost
        response = ModelResponse(token_cost=50)
        await self.clipper.after_response_async(response)
        
        self.assertEqual(self.clipper.current_max_tokens, 150)
    
    def test_on_loop_end_below_threshold(self):
        """Test on_loop_end returns None when below threshold"""
        self.clipper.current_max_tokens = 700  # 70% of 1000, below 80% threshold
        
        model_input = ModelInput(messages=[
            {"role": "user", "content": "Hello"}
        ])
        
        result = self.clipper.on_loop_end(model_input)
        
        self.assertIsNone(result)
        self.assertEqual(self.clipper.current_max_tokens, 700)
    
    def test_on_loop_end_above_threshold_no_function_calls(self):
        """Test when above threshold but no function calls to clip"""
        self.clipper.current_max_tokens = 900  # 90% of 1000, above threshold
        
        model_input = ModelInput(messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ])
        
        result = self.clipper.on_loop_end(model_input)
        
        # Should return None because no function calls to clip
        self.assertIsNone(result)
    
    def test_on_loop_end_clips_function_calls(self):
        """Test clipping function calls when above threshold"""
        self.clipper.current_max_tokens = 900  # Above threshold
        
        # Create messages with function calls
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather"}}]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 72°F"},
            {"role": "assistant", "content": "The weather is sunny!"},
            {"role": "user", "content": "What about tomorrow?"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "get_weather"}}]
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "Rainy, 65°F"}
        ]
        
        model_input = ModelInput(messages=messages)
        result = self.clipper.on_loop_end(model_input)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AgentLoopStateEdit)
        self.assertIsNotNone(result.model_input)
        
        # Should have clipped some messages
        original_count = len(messages)
        new_count = len(result.model_input.messages)
        self.assertLess(new_count, original_count)
        self.assertEqual(self.clipper.total_clipped_msgs, original_count - new_count)
    
    def test_on_loop_end_respects_keep_fc_msgs(self):
        """Test that clipping respects the keep_fc_msgs parameter"""
        clipper = RollContextClipper(
            max_context_tokens=1000,
            threshold_ratio=0.5,  # Low threshold to trigger clipping
            keep_fc_msgs=2  # Keep at least 2 function call messages
        )
        clipper.current_max_tokens = 600  # Above threshold
        
        # Create messages with multiple function call rounds
        messages = []
        for i in range(5):  # 5 rounds of function calls
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({
                "role": "assistant",
                "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "test_func"}}]
            })
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": f"Response {i}"})
        
        model_input = ModelInput(messages=messages)
        result = clipper.on_loop_end(model_input)
        
        self.assertIsNotNone(result)
        
        # Count remaining function call messages
        remaining_fc_count = 0
        for msg in result.model_input.messages:
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                remaining_fc_count += 1
            elif msg.get('role') == 'tool':
                remaining_fc_count += 1
        
        # Should have kept at least keep_fc_msgs function call messages
        self.assertGreaterEqual(remaining_fc_count, 2)
    
    @pytest.mark.asyncio
    async def test_on_exit_async_with_output_interface(self):
        """Test on_exit_async sends notification"""
        await self.clipper.bind_io(self.mock_output)
        
        # Set some clipping statistics
        self.clipper.total_clipped_msgs = 15
        
        model_input = ModelInput(messages=[{"role": "user", "content": "Test"}])
        await self.clipper.on_exit_async(model_input)
        
        # Verify notification was sent
        self.mock_output.push_message.assert_called_once()
        call_args = self.mock_output.push_message.call_args[0][0]
        self.assertIsInstance(call_args, UserNotify)
        self.assertEqual(call_args.type, 'notify')
        self.assertEqual(call_args.title, 'Context Clipped')
        self.assertIn("15 function calling messages have been removed", call_args.content)
        self.assertIn("current message list length: 1", call_args.content)
    
    @pytest.mark.asyncio
    async def test_on_exit_async_without_output_interface(self):
        """Test on_exit_async when no output interface is bound"""
        # Don't bind output interface
        self.clipper.output_interface = None
        
        model_input = ModelInput(messages=[{"role": "user", "content": "Test"}])
        await self.clipper.on_exit_async(model_input)
        
        # Should not raise any error
        
    def test_on_loop_end_clipping_iteration(self):
        """Test that clipping continues until below keep_fc_msgs or no more to clip"""
        clipper = RollContextClipper(
            max_context_tokens=1000,
            threshold_ratio=0.5,
            keep_fc_msgs=1
        )
        clipper.current_max_tokens = 600
        
        # Create many rounds of function calls
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Q{i}"})
            messages.append({
                "role": "assistant",
                "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "test"}}]
            })
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": f"A{i}"})
        
        model_input = ModelInput(messages=messages)
        result = clipper.on_loop_end(model_input)
        
        self.assertIsNotNone(result)
        
        # Should have clipped most messages, leaving only the most recent ones
        remaining_messages = result.model_input.messages
        
        # Find the most recent assistant tool call
        last_assistant_idx = None
        for idx, msg in enumerate(reversed(remaining_messages)):
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                last_assistant_idx = len(remaining_messages) - 1 - idx
                break
        
        # Should have at least one complete function call round remaining
        self.assertIsNotNone(last_assistant_idx)
        self.assertLess(last_assistant_idx, len(remaining_messages) - 1)
        self.assertEqual(remaining_messages[last_assistant_idx + 1].get('role'), 'tool')
    
    def test_deep_copy_in_clipping(self):
        """Test that original model_input is not modified"""
        self.clipper.current_max_tokens = 900
        
        original_messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test"}}]
            }
        ]
        
        model_input = ModelInput(messages=original_messages)
        result = self.clipper.on_loop_end(model_input)
        
        # Original should be unchanged
        self.assertEqual(len(model_input.messages), 2)
        
        # Result should be a new object
        self.assertIsNot(result.model_input, model_input)
        self.assertIsNot(result.model_input.messages, model_input.messages)


# Helper to run async tests
# import pytest

if __name__ == '__main__':
    unittest.main()
