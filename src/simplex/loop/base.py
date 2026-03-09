import os
import copy
import asyncio

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

import simplex.basics
import simplex.context
import simplex.models
import simplex.tools

from simplex.basics import (
    ModelInput,
    ModelResponse,
    ToolCall,
    ToolReturn,
    ToolSchema,
    ConflictError,
    EntityInitializationError
)
from simplex.context import ContextPlugin
from simplex.models import ConversationModel
from simplex.tools import ToolCollection


def call_coroutine_functions(target_list: List, name: str, *args, **kwargs) -> List[Any]:
    result_list: List = []
    for target in target_list:
        if not hasattr(target, name):
            continue
        target_function = getattr(target, name)
        if not callable(target_function) or not asyncio.iscoroutinefunction(target_function):
            continue
        try:
            result_list.append(target_function(*args, **kwargs))
        except Exception:
            raise
    return result_list

def call_functions(target_list: List, name: str, *args, **kwargs) -> List[Any]:
    result_list: List = []
    for target in target_list:
        if not hasattr(target, name):
            continue
        target_function = getattr(target, name)
        if not callable(target_function):
            continue
        try:
            result_list.append(target_function(*args, **kwargs))
        except Exception:
            raise
    return result_list

class AgentLoop(ABC):
    def __init__(
        self,
        agent_model: ConversationModel,
        *args: Any
    ) -> None:
        self.agent_model = agent_model
        self.tools_list: List[ToolCollection] = []
        self.context_list: List[ContextPlugin] = []
        self.instance_dict: Dict[str, Any] = {}

        self.iter: int = 0

        try:
            for instance in args:
                if isinstance(instance, ToolCollection):
                    self.tools_list.append(instance)
                    if instance.key in self.instance_dict:
                        raise ConflictError(f'duplicated instance key: {instance.key}')
                    self.instance_dict[instance.key] = instance
                elif isinstance(instance, ContextPlugin):
                    self.context_list.append(instance)
                    if instance.key in self.instance_dict:
                        raise ConflictError(f'duplicated instance key: {instance.key}')
                    self.instance_dict[instance.key] = instance
                else:
                    pass
            
            self.tool_mapping: Dict[str, ToolCollection] = {}
            self.tool_schemas: List[ToolSchema] = []

            for tool in self.tools_list:
                self.tool_schemas.extend(tool.get_tools())
                names = tool.get_names()
                for name in names:
                    if name in self.tool_mapping:
                        raise ConflictError(f'duplicated tool name: {name}')
                    self.tool_mapping[name] = tool
        except Exception as e:
            raise EntityInitializationError(self.__class__.__name__, e)

    def __getitem__(self, instance_key: str) -> Any:
        return self.instance_dict.get(instance_key, None)
    
    @property
    def _all_instances(self) -> List[ToolCollection | ContextPlugin]:
        return self.tools_list + self.context_list

    async def build(self) -> None:
        try:
            await asyncio.gather(*call_coroutine_functions(self._all_instances, 'build'))
        except Exception:
            raise

    async def release(self) -> None:
        try:
            await asyncio.gather(*call_coroutine_functions(self._all_instances, 'release'))
        except Exception:
            raise
    
    async def reset(self) -> None:
        try:
            await asyncio.gather(*call_coroutine_functions(self._all_instances, 'reset'))
        except Exception:
            raise
    
    async def procedure(self, max_iteration: int = 30) -> None:
        async def tool_not_exists(original_call: ToolCall):
            return ToolReturn(
                content = f'[ERROR]: Tool {original_call.name} not exists. Please try again.',
                original_call = original_call
            )
        
        async def tool_exception(original_call: ToolCall, e: Exception):
            return ToolReturn(
                content = f'[ERROR]: An exception has occurred during tool call: {e}. Please try again.',
                original_call = original_call
            )
        
        initial_prompt = ModelInput(messages = [], tools = self.tool_schemas)

        call_functions(self.context_list, 'on_start_procedure', agent = self)
        call_functions(self.context_list, 'on_process_prompt', model_input = initial_prompt, agent = self)
        call_functions(self.tools_list, 'on_init_output', model_input = initial_prompt, agent = self)
        call_functions(self.context_list, 'on_prompt_ready', model_input = initial_prompt, agent = self)

        input: ModelInput = copy.deepcopy(initial_prompt)
        output: Optional[ModelResponse] = None
        for self.iter in range(max_iteration):
            output = await self.agent_model.generate(input)
            call_functions(self.context_list, 'on_model_response', model_response = output, agent = self)
            
            if output.tool_call is not None and len(output.tool_call) > 0:
                tool_call_tasks: List = []
                for tool_call in output.tool_call:
                    if tool_call.name in self.tool_mapping:
                        try:
                            dispatched = self.tool_mapping[tool_call.name](tool_call)
                            tool_call_tasks.append(dispatched)
                        except Exception as e:
                            tool_call_tasks.append(tool_exception(tool_call, e))
                    else:
                        tool_call_tasks.append(tool_not_exists(tool_call))
                tool_returns = await asyncio.gather(*tool_call_tasks)
                call_functions(self.context_list, 'on_tool_return', tool_return = tool_returns, agent = self)
                input = self.agent_model.tool_return_integrate(input, output, tool_returns)

            if output.response != '':
                call_functions(self.context_list, 'on_final_answer', model_response = output, agent = self)
                if output.tool_call is None or len(output.tool_call) == 0:
                    break

    async def __aenter__(self):
        await self.build()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.release()
        return False

if __name__ == '__main__':
    pass
