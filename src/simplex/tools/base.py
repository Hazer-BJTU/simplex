import os
import copy
import asyncio

from typing import Dict, List
from abc import ABC, abstractmethod

import simplex.basics.exception
import simplex.basics.dataclass

from simplex.basics.exception import ParameterError
from simplex.basics.dataclass import ToolCall, ToolReturn


class ToolCollection(ABC):
    TOOL_METHOD_PREFIX = '_tool_'

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
        
        if not hasattr(self, ToolCollection.TOOL_METHOD_PREFIX + function_name):
            raise ParameterError(
                'dispatch',
                'tool_call',
                f'undefined tool_call.name = {function_name} or member {ToolCollection.TOOL_METHOD_PREFIX + function_name}',
                type_hint='ToolCall',
                class_name=self.__class__.__name__
            )
        
        target_function = getattr(self, ToolCollection.TOOL_METHOD_PREFIX + function_name)

        try:
            result_text: str = target_function(**arguments)
            return ToolReturn(result_text, copy.deepcopy(tool_call))
        except Exception:
            raise

if __name__ == '__main__':
    pass
