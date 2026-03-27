import json
import uuid
import copy

from typing import List, Dict, Optional, Any

import simplex.basics
import simplex.models.base

from simplex.basics import (
    ModelInput,
    ModelResponse,
    ToolCall,
    ToolReturn,
    RequestError,
    ParameterError,
    ToolSchema
)
from simplex.models.base import ConversationModel


def openai_compatiable_translate(model_input: ModelInput) -> Dict:
    def to_openai_function_calling_schema(tool_schema: ToolSchema) -> Dict:
        properties: Dict = {}
        for param in tool_schema.params:
            param_property: Dict[str, Any] = {
                'type': param.type,
                'description': param.description
            }
            if param.enum:
                param_property['enum'] = param.enum
            properties[param.field] = param_property

        required: List = [
            param.field
            for param in tool_schema.params
            # if param.required
        ]
 
        function_body: Dict = {
            'name': tool_schema.name,
            'description': tool_schema.description,
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required,
                "additionalProperties": False
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

class DeepSeekConversationModel(ConversationModel):
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        client_configs: Optional[Dict] = None, 
        default_generate_configs: Optional[Dict] = None, 
        instance_id: Optional[str] = None, 
        model: str = 'deepseek-reasoner',
        enable_thinking: bool = True
    ) -> None:
        super().__init__(
            base_url,
            api_key,
            client_configs if client_configs is not None else {},
            default_generate_configs if default_generate_configs is not None else {},
            instance_id if instance_id is not None else uuid.uuid4().hex
        )

        self.model = model
        self.enable_thinking = enable_thinking

        self.completion_extras = {}

        if self.enable_thinking:
            self.completion_extras['extra_body'] = {
                "thinking": {"type": "enabled"}
            }

        self._default_generate_configs['model'] = self.model

    def clone(self) -> "DeepSeekConversationModel":
        return DeepSeekConversationModel(
            self._base_url,
            self._api_key,
            self._client_configs,
            self._default_generate_configs,
            uuid.uuid4().hex,
            self.model,
            self.enable_thinking
        )
    
    async def generate(self, model_input: ModelInput) -> ModelResponse:
        try:
            assert self.client is not None
            completion = await self.client.chat.completions.create(**(
                self._default_generate_configs | 
                openai_compatiable_translate(model_input) | 
                self.completion_extras
            ), timeout = 600)
        except AssertionError:
            raise
        except Exception as e:
            raise RequestError(original = e)
        
        reasoning_content = completion.choices[0].message.reasoning_content
        content = completion.choices[0].message.content
        tool_calls = completion.choices[0].message.tool_calls

        prompt_tokens, completion_tokens = 0, 0
        prompt_cache_hit_tokens = 0
        if hasattr(completion, 'usage'):
            if hasattr(completion.usage, 'prompt_tokens'):
                prompt_tokens += completion.usage.prompt_tokens
            if hasattr(completion.usage, 'completion_tokens'):
                completion_tokens += completion.usage.completion_tokens
            if hasattr(completion.usage, 'prompt_cache_hit_tokens'):
                prompt_cache_hit_tokens += completion.usage.prompt_cache_hit_tokens

        if tool_calls:
            tool_call_objects: Optional[List[ToolCall]] = [
                ToolCall(id = tool.id, name = tool.function.name, arguments = json.loads(tool.function.arguments))
                for tool in tool_calls
            ]
        else:
            tool_call_objects: Optional[List[ToolCall]] = None

        return ModelResponse(
            response = content,
            token_cost = prompt_tokens + completion_tokens,
            reasoning_content = reasoning_content,
            tool_call = tool_call_objects,
            extras = {
                'prompt_tokens': prompt_tokens, 
                'completion_tokens': completion_tokens,
                'prompt_cache_hit_tokens': prompt_cache_hit_tokens,
                'original_message': completion.choices[0].message
            }
        )
    
    async def batch_response(self, inputs: List[ModelInput]) -> List[ModelResponse]:
        return [ ModelResponse(response='not supported yet..') ]
    
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
        
        new_input: ModelInput = copy.deepcopy(input)
        assert new_input.messages is not None
        assert response.extras is not None and 'original_message' in response.extras
        new_input.messages.append(response.extras['original_message'])
        for record in tool_return:
            new_input.messages.append({
                'role': 'tool',
                'content': record.content,
                'tool_call_id': record.id
            })

        return new_input
    
    def final_response_integrate(self, input: ModelInput, response: ModelResponse, **kwargs) -> ModelInput:
        new_input: ModelInput = copy.deepcopy(input)
        assert new_input.messages is not None
        assert response.extras is not None and 'original_message' in response.extras
        new_input.messages.append(response.extras['original_message'])

        return new_input

if __name__ == '__main__':
    pass
