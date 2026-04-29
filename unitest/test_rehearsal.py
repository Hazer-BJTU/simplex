import os
import uuid
import pathlib
import pytest
import asyncio

from pathlib import Path

import simplex.basics
import simplex.context
import simplex.models
import simplex.tools
import simplex.loop
import simplex.io

from simplex.basics import (
    PromptTemplate, 
    ModelResponse, 
    ToolCall,
    ToolReturn,
    LoopInformation,
    ModelInput,
    LogExceptionHandler,
)
from simplex.context import TrajectoryLogContext
from simplex.models import MockConversationModel
from simplex.tools import MockCalculator
from simplex.loop import AgentLoop

MODULE_PATH: Path = Path(__file__).resolve().parent
OUTPUT_PATH: Path = MODULE_PATH / 'output/test_rehearsal'


@pytest.mark.no_requirements
def test_rehearsal_basic_text_response() -> None:
    """
    Test that rehearsal mode can play back a simple text-only response
    without ever calling the real model backend.
    """
    async def test_body() -> None:
        # Create a rehearsal list with one final text response
        rehearsal_list = [
            LoopInformation(
                model_response = ModelResponse(response = 'This is a rehearsed response.')
            )
        ]

        # Mock model with empty expected_responses — should never be called
        model = MockConversationModel(
            expected_responses = [],
            delay = 0.0
        )

        async with AgentLoop(
            model, 
            LogExceptionHandler(), 
            TrajectoryLogContext(instance_id = 'rehearsal_basic')
        ) as loop:
            response = await loop.complete(
                system = PromptTemplate('You are a helpful assistant'),
                user = PromptTemplate('This is a test.'),
                rehearsal_mode = True,
                rehearsal_list = rehearsal_list
            )

            # Verify the final output contains the rehearsed response
            assert response.messages is not None
            last_msg = response.messages[-1]
            assert last_msg['role'] == 'assistant'
            assert last_msg['content'] == 'This is a rehearsed response.'

        # Verify the model was never called (iterator stays at 0)
        assert model.iterator == 0

    asyncio.run(test_body())


@pytest.mark.no_requirements
def test_rehearsal_with_tool_calls() -> None:
    """
    Test that rehearsal mode correctly drives a multi-step interaction
    involving tool calls: the rehearsed model_response with tool_calls
    triggers actual tool execution, and the final rehearsed text response
    terminates the loop.
    """
    async def test_body() -> None:
        tool_call_id_1 = uuid.uuid4().hex
        tool_call_id_2 = uuid.uuid4().hex

        # Step 1: model says "let me calculate 3+4" via tool call
        # Step 2: model gives the final answer
        rehearsal_list = [
            LoopInformation(
                model_response = ModelResponse(
                    tool_call = [
                        ToolCall(tool_call_id_1, 'calculator', {'operation': '+', 'operand1': 3.0, 'operand2': 4.0})
                    ]
                )
            ),
            LoopInformation(
                model_response = ModelResponse(
                    tool_call = [
                        ToolCall(tool_call_id_2, 'calculator', {'operation': '*', 'operand1': 2.0, 'operand2': 5.0})
                    ]
                )
            ),
            LoopInformation(
                model_response = ModelResponse(response = 'The final answer is 7 and then 10.')
            )
        ]

        # Mock model with empty expected_responses — should never be called
        model = MockConversationModel(
            expected_responses = [],
            delay = 0.0
        )

        async with AgentLoop(
            model, 
            LogExceptionHandler(),
            TrajectoryLogContext(instance_id = 'rehearsal_tool'),
            MockCalculator(ask_for_permission = False)
        ) as loop:
            response = await loop.complete(
                system = PromptTemplate('You are a helpful assistant'),
                user = PromptTemplate('Calculate 3+4 then multiply by something.'),
                rehearsal_mode = True,
                rehearsal_list = rehearsal_list,
                max_iteration = 10
            )

            # Verify the final output
            assert response.messages is not None

            # The messages should contain the tool call and tool return traces
            # Find the tool return for calculator call 1
            tool_returns_found = 0
            final_response_found = False
            for msg in response.messages:
                if msg.get('role') == 'tool' and msg.get('tool_call_id') == tool_call_id_1:
                    assert msg['content'] == '7.0'  # 3 + 4 = 7
                    tool_returns_found += 1
                if msg.get('role') == 'tool' and msg.get('tool_call_id') == tool_call_id_2:
                    assert msg['content'] == '10.0'  # 2 * 5 = 10
                    tool_returns_found += 1
                if msg.get('role') == 'assistant' and msg.get('content') == 'The final answer is 7 and then 10.':
                    final_response_found = True

            assert tool_returns_found == 2, f"Expected 2 tool returns, found {tool_returns_found}"
            assert final_response_found, "Final rehearsed response not found in messages"

        # Verify the model was never called
        assert model.iterator == 0

    asyncio.run(test_body())


@pytest.mark.no_requirements
def test_rehearsal_exhaustion_fallback() -> None:
    """
    Test that when the rehearsal list is exhausted (no more model_response
    entries), the AgentLoop falls back to calling the real model backend.
    """
    async def test_body() -> None:
        tool_call_id = uuid.uuid4().hex

        # Rehearsal list: only a single tool call step, then exhausted
        # After exhaustion, the real model takes over to provide the final answer
        rehearsal_list = [
            LoopInformation(
                model_response = ModelResponse(
                    tool_call = [
                        ToolCall(tool_call_id, 'calculator', {'operation': '+', 'operand1': 10.0, 'operand2': 20.0})
                    ]
                )
            )
        ]

        # The real model provides a final response after rehearsal exhaustion
        model = MockConversationModel(
            expected_responses = [
                ModelResponse(response = 'The answer from the real model is 30.')
            ],
            delay = 0.0
        )

        async with AgentLoop(
            model, 
            LogExceptionHandler(),
            TrajectoryLogContext(instance_id = 'rehearsal_exhaust'),
            MockCalculator(ask_for_permission = False)
        ) as loop:
            response = await loop.complete(
                system = PromptTemplate('You are a helpful assistant'),
                user = PromptTemplate('What is 10+20?'),
                rehearsal_mode = True,
                rehearsal_list = rehearsal_list,
                max_iteration = 10
            )

            assert response.messages is not None

            # Should have the tool call trace from rehearsal
            tool_found = False
            real_model_found = False
            for msg in response.messages:
                if msg.get('role') == 'tool' and msg.get('tool_call_id') == tool_call_id:
                    assert msg['content'] == '30.0'  # 10 + 20 = 30
                    tool_found = True
                if msg.get('role') == 'assistant' and msg.get('content') == 'The answer from the real model is 30.':
                    real_model_found = True

            assert tool_found, "Tool call from rehearsal step not found"
            assert real_model_found, "Real model response after exhaustion not found"

        # Verify the model WAS called (iterator advanced to 1)
        assert model.iterator == 1

    asyncio.run(test_body())


@pytest.mark.no_requirements
def test_rehearsal_empty_or_none_list() -> None:
    """
    Test that when rehearsal_mode=True but rehearsal_list is empty or None,
    the AgentLoop behaves normally (falls back to real model immediately).
    """
    async def test_body() -> None:
        model = MockConversationModel(
            expected_responses = [
                ModelResponse(response = 'Normal response from real model.')
            ],
            delay = 0.0
        )

        # Test with empty list
        async with AgentLoop(
            model.clone(), 
            LogExceptionHandler(),
            TrajectoryLogContext(instance_id = 'rehearsal_empty')
        ) as loop:
            response = await loop.complete(
                system = PromptTemplate('You are a helpful assistant'),
                user = PromptTemplate('Hello.'),
                rehearsal_mode = True,
                rehearsal_list = []
            )

            assert response.messages is not None
            last_msg = response.messages[-1]
            assert last_msg['role'] == 'assistant'
            assert last_msg['content'] == 'Normal response from real model.'

        # Test with None
        model2 = MockConversationModel(
            expected_responses = [
                ModelResponse(response = 'Also from real model.')
            ],
            delay = 0.0
        )

        async with AgentLoop(
            model2, 
            LogExceptionHandler(),
            TrajectoryLogContext(instance_id = 'rehearsal_none')
        ) as loop:
            response = await loop.complete(
                system = PromptTemplate('You are a helpful assistant'),
                user = PromptTemplate('Hello again.'),
                rehearsal_mode = True,
                rehearsal_list = None
            )

            assert response.messages is not None
            last_msg = response.messages[-1]
            assert last_msg['role'] == 'assistant'
            assert last_msg['content'] == 'Also from real model.'

    asyncio.run(test_body())


@pytest.mark.no_requirements
def test_rehearsal_skips_null_model_response() -> None:
    """
    Test that LoopInformation entries with model_response=None are skipped,
    and the next valid model_response in the list is used instead.
    """
    async def test_body() -> None:
        # Three entries: null, valid, another null (should be ignored after break)
        rehearsal_list = [
            LoopInformation(
                model_response = None,  # skipped
                tool_returns = []
            ),
            LoopInformation(
                model_response = ModelResponse(response = 'Response after skipping nulls.')
            ),
            LoopInformation(
                model_response = None  # not reached (loop breaks on text response)
            )
        ]

        model = MockConversationModel(
            expected_responses = [],
            delay = 0.0
        )

        async with AgentLoop(
            model, 
            LogExceptionHandler(),
            TrajectoryLogContext(instance_id = 'rehearsal_skip_null')
        ) as loop:
            response = await loop.complete(
                system = PromptTemplate('You are a helpful assistant'),
                user = PromptTemplate('Test skip null.'),
                rehearsal_mode = True,
                rehearsal_list = rehearsal_list
            )

            assert response.messages is not None
            last_msg = response.messages[-1]
            assert last_msg['role'] == 'assistant'
            assert last_msg['content'] == 'Response after skipping nulls.'

        # Model should not have been called
        assert model.iterator == 0

    asyncio.run(test_body())


@pytest.mark.no_requirements
def test_rehearsal_multiple_tool_calls_in_single_response() -> None:
    """
    Test that rehearsed model_responses with multiple tool_calls in one
    response are handled correctly (all tool calls executed concurrently).
    """
    async def test_body() -> None:
        call_id_1 = uuid.uuid4().hex
        call_id_2 = uuid.uuid4().hex

        rehearsal_list = [
            LoopInformation(
                model_response = ModelResponse(
                    tool_call = [
                        ToolCall(call_id_1, 'calculator', {'operation': '+', 'operand1': 5.0, 'operand2': 3.0}),
                        ToolCall(call_id_2, 'calculator', {'operation': '*', 'operand1': 10.0, 'operand2': 2.0})
                    ]
                )
            ),
            LoopInformation(
                model_response = ModelResponse(response = 'Results: 5+3=8 and 10*2=20.')
            )
        ]

        model = MockConversationModel(
            expected_responses = [],
            delay = 0.0
        )

        async with AgentLoop(
            model, 
            LogExceptionHandler(),
            TrajectoryLogContext(instance_id = 'rehearsal_multi_tool'),
            MockCalculator(ask_for_permission = False)
        ) as loop:
            response = await loop.complete(
                system = PromptTemplate('You are a helpful assistant'),
                user = PromptTemplate('Do two calculations.'),
                rehearsal_mode = True,
                rehearsal_list = rehearsal_list,
                max_iteration = 10
            )

            assert response.messages is not None

            # Both tool returns should be present
            tool_call_ids_seen = set()
            for msg in response.messages:
                if msg.get('role') == 'tool':
                    tool_call_ids_seen.add(msg['tool_call_id'])
                    if msg['tool_call_id'] == call_id_1:
                        assert msg['content'] == '8.0'
                    elif msg['tool_call_id'] == call_id_2:
                        assert msg['content'] == '20.0'

            assert call_id_1 in tool_call_ids_seen
            assert call_id_2 in tool_call_ids_seen

            # Final response present
            final_contents = [msg['content'] for msg in response.messages if msg.get('role') == 'assistant' and msg.get('content')]
            assert 'Results: 5+3=8 and 10*2=20.' in final_contents

        assert model.iterator == 0

    asyncio.run(test_body())


if __name__ == '__main__':
    pass
