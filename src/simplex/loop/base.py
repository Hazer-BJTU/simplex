import os
import asyncio

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

import simplex.basics.dataclass
import simplex.basics.exception
import simplex.context.base
import simplex.models.base
import simplex.tools.base

from simplex.basics.dataclass import ToolCall, ToolReturn, ModelInput, ModelResponse
from simplex.basics.exception import ConflictError
from simplex.context.base import ContextPlugin
from simplex.models.base import ConversationModel
from simplex.tools.base import ToolCollection


def call_coroutine_functions(target_list: List, name: str, *args, **kwargs) -> List[Any]:
    result_list: List = []
    for target in target_list:
        if not hasattr(target, name):
            continue
        target_function = getattr(target, name)
        if not callable(target_function) or not asyncio.iscoroutinefunction(target_function):
            continue
        result_list.append(target_function(*args, **kwargs))
    return result_list

def call_functions(target_list: List, name: str, *args, **kwargs) -> List[Any]:
    result_list: List = []
    for target in target_list:
        if not hasattr(target, name):
            continue
        target_function = getattr(target, name)
        if not callable(target_function):
            continue
        result_list.append(target_function(*args, **kwargs))
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
        self.tool_schemas: List[Dict] = []

        for tool in self.tools_list:
            self.tool_schemas.extend(tool.get_tools())
            names = tool.get_names()
            for name in names:
                if name in self.tool_mapping:
                    raise ConflictError(f'duplicated tool name: {name}')
                self.tool_mapping[name] = tool

    def __getitem__(self, instance_key: str) -> Any:
        return self.instance_dict.get(instance_key, None)

    async def build(self) -> None:
        await asyncio.gather(*call_coroutine_functions(
            self.tools_list + self.context_list,
            'build'
        ))
        return

    async def release(self) -> None:
        await asyncio.gather(*call_coroutine_functions(
            self.tools_list + self.context_list,
            'release'
        ))
        return
    
    async def reset(self) -> None:
        await asyncio.gather(*call_coroutine_functions(
            self.tools_list + self.context_list,
            'reset'
        ))
        return
    
    async def procedure(self) -> None:
        model_input = ModelInput(messages = [], tools = self.tool_schemas)


if __name__ == '__main__':
    '''
    from simplex.tools.pyinterpreter import PythonInterpreter
    tool = PythonInterpreter(use_container=True, default_image='python:3.11-slim')

    agent_loop = AgentLoop(None, tool, tool)
    '''
    pass
