import os
import uuid

from typing import List, Dict

import simplex.basics
import simplex.models.base

from simplex.basics import (
    ModelInput, 
    ModelResponse, 
    ToolCall, 
    ToolReturn, 
    RequestError, 
    ParameterError
)
from simplex.models.base import EmbeddingModel, ConversationModel


class MockConversationModel(ConversationModel):
    def __init__(
        self, 
        instance_id: str = uuid.uuid4().hex,
        expected_response: List[ModelResponse] = []
    ) -> None:
        super().__init__('', '', {}, {}, instance_id, True)
        
        self.expected_response = expected_response
        
        self.iterator: int = 0

    async def generate(self, model_input: ModelInput) -> ModelResponse:
        return ModelResponse()
    
    async def batch_response(self, inputs: List[ModelInput]) -> List[ModelResponse]:
        return []
    
    def tool_return_integrate(self, input: ModelInput, response: ModelResponse, tool_return: List[ToolReturn], **kwargs) -> ModelInput:
        return ModelInput()

if __name__ == '__main__':
    pass
