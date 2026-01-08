import os
import asyncio

from typing import Dict, List
from abc import ABC, abstractmethod

import simplex.basics.exception
import simplex.basics.dataclass

from simplex.basics.dataclass import ToolCall, ToolReturn
from simplex.basics.exception import ParameterError, ImplementationError


class ToolCollection(ABC):
    def __init__(self, name_mapping: Dict) -> None:
        self.name_mapping: Dict = name_mapping

    async def __call__(self, tool_call: ToolCall) -> ToolReturn:
        return await self.dispatch(tool_call)

    @abstractmethod
    async def build(self) -> None:
        pass

    @abstractmethod
    async def release(self) -> None:
        pass

    @abstractmethod
    def get_tools(self) -> List[Dict]:
        pass

    @abstractmethod
    def tools_descriptions(self) -> List[Dict]:
        pass
    
    @abstractmethod
    async def dispatch(self, tool_call: ToolCall) -> ToolReturn:
        function_name: str = tool_call.name
        arguments: Dict = tool_call.arguments

        try:
            member_name: str = self.name_mapping[function_name]
        except KeyError:
            raise ParameterError(
                'dispatch',
                'tool_call',
                f'tool_call.name should be one of {str(self.name_mapping)}',
                type_hint='ToolCall',
                class_name=self.__class__.__name__
            )
        
        if not hasattr(self, member_name):
            raise ParameterError(
                'dispatch',
                'tool_call',
                f'undefined tool_call.name = {function_name} or member {member_name}',
                type_hint='ToolCall',
                class_name=self.__class__.__name__
            )
        
        target_function = getattr(self, member_name)
        if not asyncio.iscoroutinefunction(target_function):
            raise ImplementationError(
                target_function.__name__ if hasattr(target_function, '__name__') else 'unknown_function',
                'target member should be a coroutine function',
                class_name=self.__class__.__name__
            )

        try:
            result_text: str = await target_function(**arguments)
        except Exception:
            raise
        return ToolReturn(result_text, tool_call)

if __name__ == '__main__':
    pass
