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
from simplex.tools import to_openai_function_calling_schema


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
        self.base_url = base_url
        self.api_key = api_key
        self.client_configs = client_configs
        self.default_generate_configs = default_generate_configs
        self.instance_id = instance_id
        self.disable_openai_backend = disable_openai_backend
        self.translator = OpenaiTranslator()

        try:
            if not self.disable_openai_backend:
                self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
            else:
                self.client = None
        except Exception as e:
            raise EntityInitializationError(self.__class__.__name__, e)
        
    @property
    def key(self) -> str:
        return self.instance_id
        
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

class Translator(ABC):
    @abstractmethod
    def __call__(self, input: ModelInput) -> Any:
        pass

class OpenaiTranslator(Translator):
    def __call__(self, input: ModelInput) -> Dict:
        input_dict: Dict = input.to_dict()
        tools: List = input_dict.get('tools', [])
        translated_tools: List = []
        for tool in tools:
            if isinstance(tool, ToolSchema):
                translated_tools.append(to_openai_function_calling_schema(tool))
            else:
                translated_tools.append(tool)
        input_dict['tools'] = translated_tools
        return input_dict

if __name__ == '__main__':
    pass
