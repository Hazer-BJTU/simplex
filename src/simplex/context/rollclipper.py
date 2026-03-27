import os
import copy
import uuid
import asyncio

from typing import Any, List, Dict, Optional, Set, Callable, Tuple, TYPE_CHECKING

import simplex.basics
import simplex.context.base

from simplex.basics import (
    ModelResponse, 
    ModelInput,
    UserNotify,
    AgentLoopStateEdit
)
from simplex.context.base import ContextPlugin

if TYPE_CHECKING:
    from simplex.io.base import UserOutputInterface


def identify_openai_function_calling(messages: List[Dict]) -> Tuple[int, Set[int]]:
    """
    Identify function call messages in OpenAI-style conversation messages.
    
    This function scans through a list of conversation messages and identifies messages
    that are part of function/tool calling patterns. It detects:
    1. Assistant messages that contain tool_calls (function call requests)
    2. Tool messages that correspond to those tool calls (function responses)
    
    Args:
        messages: List of message dictionaries in OpenAI format. Each message should
                  have a 'role' field ('user', 'assistant', 'tool', etc.) and may
                  contain 'tool_calls' for assistant messages or 'tool_call_id' for
                  tool messages.
    
    Returns:
        A tuple containing:
        - fc_msg_cnt: Total count of function calling related messages (assistant with
                      tool_calls + tool role messages)
        - indices_to_clip: Set of message indices that can be safely removed while
                          preserving at least one complete function call round.
                          Includes the first assistant tool call message and all
                          corresponding tool response messages.
    
    Note:
        The function assumes tool call IDs are consistent across assistant and tool
        messages. It tracks the first occurrence of tool calls to ensure at least
        one complete function call round is preserved in the clipping process.
    """
    fc_msg_cnt: int = 0
    tool_call_ids: Set[str] = set()
    tool_call_indices: Set[int] = set()
    first_tool_call: Optional[int] = None
    
    # Iterate through all messages to identify function call patterns
    for idx, message in enumerate(messages):
        # Detect the first assistant message with tool_calls
        if (message.get('role', '') == 'assistant' and message.get('tool_calls', [])) and first_tool_call is None:
            first_tool_call = idx
            # Collect all tool call IDs from this assistant message
            for call in message['tool_calls']:
                if hasattr(call, 'id'):
                    tool_call_ids.add(getattr(call, 'id'))
                elif isinstance(call, dict) and 'id' in call:
                    tool_call_ids.add(call['id'])

        # Identify tool messages that correspond to previously seen tool call IDs
        if message.get('role', '') == 'tool' and message.get('tool_call_id', '') in tool_call_ids:
            tool_call_indices.add(idx)

        # Count all function calling related messages (assistant with tool_calls or tool role)
        if (message.get('role', '') == 'assistant' and message.get('tool_calls', [])) or message.get('role', '') == 'tool':
            fc_msg_cnt += 1

    # If we found a tool call, include its index in the set of indices to clip
    if first_tool_call:
        return fc_msg_cnt, {first_tool_call} | tool_call_indices
    else:
        # No tool calls found, return empty set
        return fc_msg_cnt, tool_call_indices  # fc_msg_cnt should equal to zero

class RollContextClipper(ContextPlugin):
    """
    A context plugin that clips conversation history when token usage exceeds a threshold.
    
    This plugin monitors token usage during agent execution and clips function calling
    messages from the conversation history when the token count exceeds a configurable
    threshold. It preserves a minimum number of function calling messages to maintain
    context coherence while reducing token consumption.
    
    The clipping algorithm:
    1. Tracks maximum token usage across agent responses (after_response_async)
    2. At the end of each loop iteration (on_loop_end), checks if token usage exceeds
       threshold_ratio * max_context_tokens
    3. If threshold exceeded, iteratively removes function calling message rounds
       (assistant tool calls + corresponding tool responses) while preserving at least
       keep_fc_msgs function calling messages
    4. Notifies the user of clipping statistics when the loop exits (on_exit_async)
    
    Attributes:
        max_context_tokens (int): Maximum allowed context tokens (default: 128000)
        threshold_ratio (float): Ratio of max_context_tokens at which clipping triggers
                                 (default: 0.65)
        keep_fc_msgs (int): Minimum number of function calling messages to preserve
                            (default: 50)
        identify_function (Callable): Function that identifies function call messages
        current_max_tokens (int): Maximum token count observed in current loop
        total_clipped_msgs (int): Total number of messages clipped across loops
        output_interface (Optional[UserOutputInterface]): Interface for user notifications
    """
    def __init__(
        self, 
        instance_id: Optional[str] = None,
        max_context_tokens: int = 128000,
        threshold_ratio: float = 0.65,
        keep_fc_msgs: int = 50,
        identify_function: Callable[[List[Dict]], Tuple[int, Set[int]]] = identify_openai_function_calling
    ) -> None:
        """
        Initialize the RollContextClipper with configuration parameters.
        
        Args:
            instance_id: Optional unique identifier for this plugin instance. If not
                         provided, a UUID will be generated.
            max_context_tokens: Maximum context window size in tokens. Clipping will
                                be considered when token usage exceeds
                                threshold_ratio * max_context_tokens.
            threshold_ratio: Ratio (0.0 to 1.0) of max_context_tokens that triggers
                             clipping. For example, 0.65 means clipping occurs when
                             token usage exceeds 65% of max_context_tokens.
            keep_fc_msgs: Minimum number of function calling messages to preserve
                          during clipping. This ensures some context is retained.
            identify_function: Function that identifies function call messages in a
                               conversation history. Defaults to 
                               identify_openai_function_calling which works with
                               OpenAI-style tool calling format.
        """
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex)

        self.max_context_tokens = max_context_tokens
        self.threshold_ratio = threshold_ratio
        self.keep_fc_msgs = keep_fc_msgs
        self.identify_function = identify_function

        self.current_max_tokens: int = 0
        self.total_clipped_msgs: int = 0
        self.output_interface: Optional["UserOutputInterface"] = None

    async def bind_io(self, output_interface: "UserOutputInterface", **kwargs) -> None:
        """
        Bind an output interface for sending user notifications.
        
        This method is called by the agent loop to provide a way for the plugin
        to send notifications to the user about clipping operations.
        
        Args:
            output_interface: UserOutputInterface instance for sending notifications.
            **kwargs: Additional keyword arguments (unused but preserved for compatibility).
        """
        self.output_interface = output_interface

    async def start_loop_async(self, **kwargs) -> None:
        """
        Reset tracking metrics at the start of each agent loop iteration.
        
        This method is called when a new agent loop iteration begins. It resets
        the current_max_tokens and total_clipped_msgs counters to prepare for
        tracking token usage in the new iteration.
        
        Args:
            **kwargs: Additional keyword arguments (unused but preserved for compatibility).
        """
        self.current_max_tokens = 0
        self.total_clipped_msgs = 0

    async def after_response_async(self, model_response: ModelResponse, **kwargs) -> None:
        """
        Update the maximum token usage based on the latest model response.
        
        This method is called after each model response to track the highest
        token cost encountered during the current loop iteration. The maximum
        token count is used to determine when clipping should occur.
        
        Args:
            model_response: The ModelResponse object containing token_cost information.
            **kwargs: Additional keyword arguments (unused but preserved for compatibility).
        """
        self.current_max_tokens = max(self.current_max_tokens, model_response.token_cost)

    def on_loop_end(self, model_input: ModelInput, **kwargs) -> Optional[AgentLoopStateEdit]:
        """
        Clip conversation history if token usage exceeds the configured threshold.
        
        This method is called at the end of each agent loop iteration. If the
        maximum token usage observed during the iteration exceeds the threshold
        (threshold_ratio * max_context_tokens), it performs iterative clipping
        of function calling messages while preserving at least keep_fc_msgs
        function calling messages.
        
        Args:
            model_input: The current ModelInput containing conversation messages.
            **kwargs: Additional keyword arguments (unused but preserved for compatibility).
        
        Returns:
            AgentLoopStateEdit with clipped messages if clipping occurred,
            or None if no clipping was needed (token usage below threshold).
        
        The clipping algorithm:
        1. Check if current_max_tokens >= threshold_ratio * max_context_tokens
        2. If yes, create a deep copy of the message list to avoid modifying original
        3. While there are messages left:
           a. Identify function call messages and their indices using identify_function
           b. If no function calls found, stop clipping
           c. Remove messages at identified indices
           d. Update total_clipped_msgs counter
           e. If remaining function call messages <= keep_fc_msgs, stop clipping
        4. Reset current_max_tokens for next iteration
        5. Return AgentLoopStateEdit with clipped messages
        """
        # Check if token usage exceeds threshold
        if self.current_max_tokens >= self.max_context_tokens * self.threshold_ratio:
            # Create a copy to avoid modifying the original input
            new_model_input = copy.deepcopy(model_input)
            
            # Iteratively clip function call messages until conditions met
            while new_model_input.messages:
                # Identify function call messages in current message list
                fc_msg_cnt, indices_to_clip = self.identify_function(new_model_input.messages)
                
                # Stop if no function calls found or no indices to clip
                if fc_msg_cnt == 0 or len(indices_to_clip) == 0:
                    break
                
                # Remove messages at the identified indices
                new_messages = [msg for idx, msg in enumerate(new_model_input.messages) 
                                if idx not in indices_to_clip]
                new_model_input.messages = new_messages
                self.total_clipped_msgs += len(indices_to_clip)
                
                # Stop if we've preserved enough function calling messages
                if fc_msg_cnt - len(indices_to_clip) <= self.keep_fc_msgs:
                    break
            
            # Reset token tracking for next iteration
            self.current_max_tokens = 0
            return AgentLoopStateEdit(model_input = new_model_input)
        
        # Return None if no clipping occurred (token usage below threshold)
        
    async def on_exit_async(self, model_input: ModelInput, **kwargs) -> None:
        """
        Notify the user about clipping statistics when the agent loop exits.
        
        This method is called when the agent loop is exiting. If an output
        interface is bound, it sends a notification summarizing the total
        number of function calling messages clipped and the current message
        list length.
        
        Args:
            model_input: The final ModelInput containing conversation messages.
            **kwargs: Additional keyword arguments (unused but preserved for compatibility).
        """
        if self.output_interface:
            current_length: int = len(model_input.messages) if model_input.messages else 0
            notify_content: str = f"{self.total_clipped_msgs} function calling messages have been removed, current message list length: {current_length}"
            await self.output_interface.push_message(UserNotify('notify', title = 'Context Clipped', content = notify_content))
    
if __name__ == '__main__':
    pass
