import json
import uuid
import copy

from typing import List, Dict

import simplex.basics.exception
import simplex.basics.dataclass
import simplex.models.base

from simplex.basics.exception import RequestError, ParameterError
from simplex.models.base import EmbeddingModel, ConversationModel
from simplex.basics.dataclass import ModelInput, ModelResponse, ToolCall, ToolReturn


class QwenConversationModel(ConversationModel):
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        client_configs: Dict = {}, 
        default_generate_configs: Dict = {}, 
        instance_id: str = uuid.uuid4().hex,
        qwen_model: str = 'qwen-plus',
        enable_thinking: bool = True,
        thinking_budget: int = 1024
    ) -> None:
        super().__init__(base_url, api_key, client_configs, default_generate_configs, instance_id)
        
        self.qwen_model = qwen_model
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.default_generate_configs['model'] = self.qwen_model
        self.completion_extras = {
            'stream': True, 
            'stream_options': {
                'include_usage': True
            }
        }
        if self.enable_thinking:
            self.completion_extras['extra_body'] = {
                'enable_thinking': self.enable_thinking,
                'thinking_budget': self.thinking_budget
            }

    async def generate(self, model_input: ModelInput) -> ModelResponse:
        try:
            completion = await self.client.chat.completions.create(**(self.default_generate_configs | model_input.dict | self.completion_extras))
        except Exception as e:
            raise RequestError(e)

        response: str = ''
        reasoning_content: str = ''
        tool_call: List[Dict] = []
        token_usages: List = []

        async for chunk in completion:
            if not chunk.choices:
                token_usages.append(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                elif delta.content is not None:
                    response += delta.content
                elif delta.tool_calls is not None:
                    for call in delta.tool_calls:
                        index = call.index
                        while len(tool_call) <= index:
                            tool_call.append({})
                        if call.id:
                            tool_call[index]['id'] = tool_call[index].get('id', '') + call.id
                        if call.function and call.function.name:
                            tool_call[index]['name'] = tool_call[index].get('name', '') + call.function.name
                        if call.function and call.function.arguments:
                            tool_call[index]['arguments'] = tool_call[index].get('arguments', '') + call.function.arguments
                else:
                    pass

        prompt_tokens, completion_tokens = 0, 0
        for record in token_usages:
            if hasattr(record, 'prompt_tokens'):
                prompt_tokens += record.prompt_tokens
            if hasattr(record, 'completion_tokens'):
                completion_tokens += record.completion_tokens

        tool_call_objects: List[ToolCall] = [
            ToolCall(id=record['id'], name=record['name'], arguments=json.loads(record['arguments']))
            for record in tool_call
        ]

        return ModelResponse(
            response=response,
            token_cost=prompt_tokens + completion_tokens,
            reasoning_content=reasoning_content,
            tool_call=tool_call_objects,
            extras={
                'prompt_tokens': prompt_tokens, 
                'completion_tokens': completion_tokens
            }
        )

    async def batch_response(self, inputs: List[ModelInput]) -> ModelResponse:
        return ModelResponse(response='not supported yet..')
    
    def tool_return_integrate(self, input: ModelInput, response: ModelResponse, tool_return: List[ToolReturn], **kwargs) -> ModelInput:
        if response.tool_call is None:
            raise ParameterError(
                'tool_return_integrate', 
                'response', 
                'response.tool_call should not be None',
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
            'content': '',
            'refusal': None,
            'role': 'assistant',
            'audio': None,
            'function_call': None,
            'tool_calls': [
                {
                    'id': record.id,
                    'function': {
                        'arguments': json.dumps(record.arguments),
                        'name': record.name
                    },
                    'type': 'function',
                    'index': index
                }
                for index, record in enumerate(response.tool_call)
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


if __name__ == '__main__':
    pass
