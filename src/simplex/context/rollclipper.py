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
    fc_msg_cnt: int = 0
    tool_call_ids: Set[str] = set()
    tool_call_indices: Set[int] = set()
    first_tool_call: Optional[int] = None
    for idx, message in enumerate(messages):
        if (message.get('role', '') == 'assistant' and message.get('tool_calls', [])) and first_tool_call is None:
            first_tool_call = idx
            for call in message['tool_calls']:
                if hasattr(call, 'id'):
                    tool_call_ids.add(getattr(call, 'id'))
                elif isinstance(call, dict) and 'id' in call:
                    tool_call_ids.add(call['id'])

        if message.get('role', '') == 'tool' and message.get('tool_call_id', '') in tool_call_ids:
            tool_call_indices.add(idx)

        if (message.get('role', '') == 'assistant' and message.get('tool_calls', [])) or message.get('role', '') == 'tool':
            fc_msg_cnt += 1

    if first_tool_call:
        return fc_msg_cnt, { first_tool_call } | tool_call_indices
    else:
        return fc_msg_cnt, tool_call_indices # fc_msg_cnt should equal to zero

class RollContextClipper(ContextPlugin):
    def __init__(
        self, 
        instance_id: Optional[str] = None,
        max_context_tokens: int = 128000,
        threshold_ratio: float = 0.8,
        keep_fc_msgs: int = 24,
        identify_function: Callable[[List[Dict]], Tuple[int, Set[int]]] = identify_openai_function_calling
    ) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex)

        self.max_context_tokens = max_context_tokens
        self.threshold_ratio = threshold_ratio
        self.keep_fc_msgs = keep_fc_msgs
        self.identify_function = identify_function

        self.current_max_tokens: int = 0
        self.total_clipped_msgs: int = 0
        self.output_interface: Optional["UserOutputInterface"] = None

    async def bind_io(self, output_interface: "UserOutputInterface", **kwargs) -> None:
        self.output_interface = output_interface

    async def start_loop_async(self, **kwargs) -> None:
        self.current_max_tokens = 0
        self.total_clipped_msgs = 0

    async def after_response_async(self, model_response: ModelResponse, **kwargs) -> None:
        self.current_max_tokens = max(self.current_max_tokens, model_response.token_cost)

    def on_loop_end(self, model_input: ModelInput, **kwargs) -> Optional[AgentLoopStateEdit]:
        if self.current_max_tokens >= self.max_context_tokens * self.threshold_ratio:
            new_model_input = copy.deepcopy(model_input)
            while new_model_input.messages:
                fc_msg_cnt, indices_to_clip = self.identify_function(new_model_input.messages)
                if fc_msg_cnt == 0 or len(indices_to_clip) == 0:
                    break

                new_messages = [msg for idx, msg in enumerate(new_model_input.messages) if idx not in indices_to_clip]
                new_model_input.messages = new_messages
                self.total_clipped_msgs += len(indices_to_clip)
                if fc_msg_cnt - len(indices_to_clip) <= self.keep_fc_msgs:
                    break

            self.current_max_tokens = 0
            return AgentLoopStateEdit(model_input = new_model_input)
        
    async def on_exit_async(self, model_input: ModelInput, **kwargs) -> None:
        if self.output_interface:
            current_length: int = len(model_input.messages) if model_input.messages else 0
            notify_content: str = f"{self.total_clipped_msgs} function calling messages have been removed, current message list length: {current_length}"
            await self.output_interface.push_message(UserNotify('notify', title = 'Context Clipped', content = notify_content))
    
if __name__ == '__main__':
    pass
