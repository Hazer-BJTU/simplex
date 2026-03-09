import os
import uuid
import yaml
import pathlib
import inspect
import asyncio

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

import simplex.basics

from simplex.basics import (
    ModelInput,
    ToolCall,
    ToolReturn,
    ToolSchema,
    ParameterError,
    ImplementationError
)

if TYPE_CHECKING:
    import simplex.loop

    from simplex.loop import AgentLoop


MODULE_PATH: Path = Path(__file__).resolve().parent
SCHEMA_DIR: Path = MODULE_PATH / 'schema'

class ToolCollection(ABC):
    def __init__(
        self,
        instance_id: str,
        name_mapping: Dict
    ) -> None:
        self.instance_id = instance_id
        self.name_mapping: Dict = name_mapping

    @property
    def key(self) -> str:
        return self.instance_id

    async def __call__(self, tool_call: ToolCall) -> ToolReturn:
        return await self.dispatch(tool_call)
    
    async def __aenter__(self):
        await self.build()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()
        return False

    @abstractmethod
    async def build(self) -> None:
        pass

    @abstractmethod
    async def release(self) -> None:
        pass

    @abstractmethod
    async def reset(self) -> None:
        pass

    @abstractmethod
    def get_names(self) -> List[str]:
        pass

    @abstractmethod
    def get_tools(self) -> List[Dict]:
        pass

    @abstractmethod
    def tools_descriptions(self) -> List[Dict]:
        pass

    @abstractmethod
    def on_init_output(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        pass 
    
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
        if not callable(target_function) or not inspect.iscoroutinefunction(target_function):
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

def load_tool_definitions(file_name: str) -> str:
    try:
        with open(SCHEMA_DIR / f"{file_name}.yml", 'r', encoding = 'utf8') as schema_file:
            content: str = schema_file.read()
        return content.strip()
    except Exception:
        raise

def load_schema(file_name: str, tool_name: str, rename: Optional[str] = None) -> ToolSchema:
    try:
        with open(SCHEMA_DIR / f"{file_name}.yml", 'r', encoding = 'utf8') as schema_file:
            content = yaml.safe_load(schema_file)
        tool_schema: Dict = content.get(tool_name, {})
        if rename is not None:
            name: str = rename
        else:
            name: str = tool_schema.get('name', tool_name)
        description: str = tool_schema.get('description', '')
        params: List[Dict] = tool_schema.get('params', [])
        extras: Dict = tool_schema.get('extras', {})
        formated_params: List[ToolSchema.Parameter] = []
        for param in params:
            param_field: str = param.get('field', 'unknown')
            param_type: str = param.get('type', 'string')
            param_description: str = param.get('description', '')
            param_required: bool = param.get('required', True)
            param_extras: Dict = param.get('extras', {})
            formated_params.append(ToolSchema.Parameter(
                field = param_field,
                type = param_type,
                description = param_description,
                required = param_required,
                extras = param_extras
            ))
        return ToolSchema(
            name = name,
            description = description,
            params = formated_params,
            extras = extras
        )
    except Exception:
        raise

def to_openai_function_calling_schema(tool_schema: ToolSchema) -> Dict:
    properties: Dict = {
        param.field: {
            'type': param.type,
            'description': param.description
        }
        for param in tool_schema.params
    }
    required: List = [
        param.field
        for param in tool_schema.params
        if param.required
    ]

    function_body: Dict = {
        'name': tool_schema.name,
        'description': tool_schema.description,
        'parameters': {
            'type': 'object',
            'properties': properties,
            'required': required
        }
    }

    return {
        'type': 'function',
        'function': function_body
    }

if __name__ == '__main__':
    pass
