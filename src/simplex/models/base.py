import os
import uuid

from typing import Dict, List
from openai import AsyncOpenAI
from abc import ABC, abstractmethod

import simplex.basics

from simplex.basics import (
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
        instance_id: str = uuid.uuid4().hex
    ) -> None:
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.client_configs = client_configs
        self.default_generate_configs = default_generate_configs
        self.instance_id = instance_id

        try:
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
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
        instance_id: str = uuid.uuid4().hex
    ) -> None:
        super().__init__(base_url, api_key, client_configs, default_generate_configs, instance_id)

    async def build(self) -> None:
        return
    
    async def release(self) -> None:
        return
    
    async def reset(self) -> None:
        return

    async def generate(self, model_input: ModelInput) -> ModelResponse:
        return ModelResponse()
    
    @abstractmethod
    async def batch_embedding(self, documents: List[DocumentEntry]) -> ModelResponse:
        pass
    
class ConversationModel(BaseModel):
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        client_configs: Dict = {}, 
        default_generate_configs: Dict = {},
        instance_id: str = uuid.uuid4().hex
    ) -> None:
        super().__init__(base_url, api_key, client_configs, default_generate_configs, instance_id)

    async def build(self) -> None:
        return

    async def release(self) -> None:
        return

    async def reset(self) -> None:
        return   

    async def generate(self, model_input: ModelInput) -> ModelResponse:
        return ModelResponse()

    @abstractmethod
    async def batch_response(self, inputs: List[ModelInput]) -> ModelResponse:
        pass

    @abstractmethod
    def tool_return_integrate(self, input: ModelInput, response: ModelResponse, tool_return: List[ToolReturn], **kwargs) -> ModelInput:
        pass

if __name__ == '__main__':
    pass
