import os
import uuid

from typing import Dict, List, Any
from openai import AsyncOpenAI
from abc import ABC, abstractmethod

import simplex.basics
import simplex.tools

from simplex.basics import (
    ToolSchema,
    ModelInput, 
    ModelResponse, 
    DocumentEntry, 
    ToolReturn, 
    EntityInitializationError
)


class BaseModel(ABC):
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        client_configs: Dict = {}, 
        default_generate_configs: Dict = {},
        instance_id: str = uuid.uuid4().hex,
        disable_openai_backend: bool = False
    ) -> None:
        super().__init__()
        self.__base_url = base_url
        self.__api_key = api_key
        self.__client_configs = client_configs
        self.__default_generate_configs = default_generate_configs
        self.__instance_id = instance_id
        self.__disable_openai_backend = disable_openai_backend

        try:
            if not self.__disable_openai_backend:
                self.client = AsyncOpenAI(
                    base_url = self.__base_url, 
                    api_key = self.__api_key,
                    **self.__client_configs
                )
            else:
                self.client = None
        except Exception as e:
            raise EntityInitializationError(self.__class__.__name__, e)
        
    @property
    def key(self) -> str:
        return self.__instance_id
    
    @property
    def _base_url(self) -> str:
        return self.__base_url
    
    @property
    def _api_key(self) -> str:
        return self.__api_key
    
    @property
    def _client_configs(self) -> Dict:
        return self.__client_configs
    
    @property
    def _default_generate_configs(self) -> Dict:
        return self.__default_generate_configs
    
    @abstractmethod
    def clone(self) -> "BaseModel":
        pass
        
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
    async def generate(self, model_input: ModelInput) -> ModelResponse:
        pass

class EmbeddingModel(BaseModel):
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        client_configs: Dict = {}, 
        default_generate_configs: Dict = {},
        instance_id: str = uuid.uuid4().hex,
        disable_openai_backend: bool = False
    ) -> None:
        super().__init__(
            base_url, 
            api_key, 
            client_configs, 
            default_generate_configs, 
            instance_id,
            disable_openai_backend
        )

    def clone(self) -> "EmbeddingModel":
        return None # type: ignore

    async def build(self) -> None:
        return
    
    async def release(self) -> None:
        return
    
    async def reset(self) -> None:
        return

    async def generate(self, model_input: ModelInput) -> ModelResponse:
        return ModelResponse()
    
    @abstractmethod
    async def batch_embedding(self, documents: List[DocumentEntry]) -> List[ModelResponse]:
        pass
    
class ConversationModel(BaseModel):
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        client_configs: Dict = {}, 
        default_generate_configs: Dict = {},
        instance_id: str = uuid.uuid4().hex,
        disable_openai_backend: bool = False
    ) -> None:
        super().__init__(
            base_url, 
            api_key, 
            client_configs, 
            default_generate_configs, 
            instance_id,
            disable_openai_backend
        )

    def clone(self) -> "ConversationModel":
        return None # type: ignore

    async def build(self) -> None:
        return

    async def release(self) -> None:
        return

    async def reset(self) -> None:
        return   

    async def generate(self, model_input: ModelInput) -> ModelResponse:
        return ModelResponse()

    @abstractmethod
    async def batch_response(self, inputs: List[ModelInput]) -> List[ModelResponse]:
        pass

    @abstractmethod
    def tool_return_integrate(self, input: ModelInput, response: ModelResponse, tool_return: List[ToolReturn], **kwargs) -> ModelInput:
        pass

    @abstractmethod
    def final_response_integrate(self, input: ModelInput, response: ModelResponse, **kwargs) -> ModelInput:
        pass

def openai_compatiable_translate(model_input: ModelInput) -> Dict:
    def to_openai_function_calling_schema(tool_schema: ToolSchema) -> Dict:
        properties: Dict = {
            param.field: {
                'type': param.type,
                'description': param.description
            } for param in tool_schema.params
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

    output_dict: Dict = {}
    if model_input.model is not None:
        output_dict['model'] = model_input.model
    if model_input.messages is not None:
        output_dict['messages'] = model_input.messages
    if model_input.tools is not None:
        output_dict['tools'] = [
            to_openai_function_calling_schema(schema)
            for schema in model_input.tools
        ]
    if model_input.input is not None:
        output_dict['input'] = model_input.input
    if model_input.extras is not None:
        output_dict |= model_input.extras
    return output_dict

if __name__ == '__main__':
    pass
