import os
import json
import copy
import uuid

from typing import Optional, List, Dict, Callable

import simplex.basics
import simplex.models.base

from simplex.basics import (
    ModelInput, 
    ModelResponse, 
    ToolCall, 
    ToolReturn, 
    ParameterError,
    RuntimeError
)
from simplex.models.base import EmbeddingModel, ConversationModel


class MockConversationModel(ConversationModel):
    def __init__(
        self, 
        instance_id: str = uuid.uuid4().hex,
        generator: Optional[Callable[[ModelInput], ModelResponse]] = None,
        expected_responses: List[ModelResponse] = [],
        cyclic: bool = True
    ) -> None:
        super().__init__('', '', {}, {}, instance_id, True)
        
        self.generator = generator
        self.expected_responses = expected_responses
        self.cyclic = cyclic
        
        self.length: int = len(expected_responses)
        self.iterator: int = 0

    async def generate(self, model_input: ModelInput) -> ModelResponse:
        if self.generator is not None:
            try:
                return self.generator(model_input)
            except Exception:
                raise
        else:
            if self.iterator >= self.length:
                if self.cyclic:
                    self.iterator = 0
                else:
                    raise RuntimeError(content = f"{self.__class__.__name__} has run out of expected responses")
            response = self.expected_responses[self.iterator]
            self.iterator += 1
            return response
    
    async def batch_response(self, inputs: List[ModelInput]) -> List[ModelResponse]:
        if self.generator is not None:
            try:
                return [ self.generator(model_input) for model_input in inputs ]
            except Exception:
                raise
        else:
            responses: List[ModelResponse] = []
            for _ in inputs:
                if self.iterator >= self.length:
                    if self.cyclic:
                        self.iterator = 0
                    else:
                        raise RuntimeError(content = f"{self.__class__.__name__} has run out of expected responses")
                response = self.expected_responses[self.iterator]
                self.iterator += 1
                responses.append(response)
            return responses
    
    def tool_return_integrate(self, input: ModelInput, response: ModelResponse, tool_return: List[ToolReturn], **kwargs) -> ModelInput:
        if response.tool_call is None or len(response.tool_call) == 0:
            raise ParameterError(
                'tool_return_integrate', 
                'response', 
                'response.tool_call should not be empty',
                type_hint='ModelResponse',
                class_name=self.__class__.__name__
            )
        
        if input.messages is None:
            raise ParameterError(
                'tool_return_integrate',
                'input',
                'input.messages should not be None',
                type_hint='ModelInput',
                class_name=self.__class__.__name__
            )
        
        assistant_tool_call_template: Dict = {
            'role': 'assistant',
            'tool_calls': [
                {
                    'name': record.name,
                    'arguments': json.dumps(record.arguments),
                    'tool_call_id': record.id
                }
                for record in response.tool_call
            ]
        }

        new_input: ModelInput = copy.deepcopy(input)
        assert new_input.messages is not None
        new_input.messages.append(assistant_tool_call_template)
        for record in tool_return:
            new_input.messages.append({
                'role': 'tool',
                'content': record.content,
                'tool_call_id': record.id
            })

        return new_input
    
    def set_generator(self, generator: Callable[[ModelInput], ModelResponse]) -> None:
        self.generator = generator
        return
    
    def set_expected(self, expected_responses: List[ModelResponse]) -> None:
        self.expected_responses = expected_responses
        self.length = len(self.expected_responses)
        self.iterator = 0
        return

if __name__ == '__main__':
    pass
