import os
import copy
import asyncio

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

import simplex.basics.dataclass
import simplex.basics.exception
import simplex.context.base
import simplex.models.base
import simplex.tools.base

from simplex.basics.dataclass import ToolCall, ToolReturn, ModelInput, ModelResponse
from simplex.basics.exception import ConflictError, RequestError
from simplex.context.base import ContextPlugin
from simplex.models.base import ConversationModel
from simplex.tools.base import ToolCollection


def call_coroutine_functions(target_list: List, name: str, *args, **kwargs) -> List[Any]:
    result_list: List = []
    for target in target_list:
        if not hasattr(target, name):
            continue
        target_function = getattr(target, name)
        if not callable(target_function) or not asyncio.iscoroutinefunction(target_function):
            continue
        result_list.append(target_function(*args, **kwargs))
    return result_list

def call_functions(target_list: List, name: str, *args, **kwargs) -> List[Any]:
    result_list: List = []
    for target in target_list:
        if not hasattr(target, name):
            continue
        target_function = getattr(target, name)
        if not callable(target_function):
            continue
        result_list.append(target_function(*args, **kwargs))
    return result_list

class AgentLoop(ABC):
    def __init__(
        self,
        agent_model: ConversationModel,
        *args: Any
    ) -> None:
        self.agent_model = agent_model
        self.tools_list: List[ToolCollection] = []
        self.context_list: List[ContextPlugin] = []
        self.instance_dict: Dict[str, Any] = {}

        for instance in args:
            if isinstance(instance, ToolCollection):
                self.tools_list.append(instance)
                if instance.key in self.instance_dict:
                    raise ConflictError(f'duplicated instance key: {instance.key}')
                self.instance_dict[instance.key] = instance
            elif isinstance(instance, ContextPlugin):
                self.context_list.append(instance)
                if instance.key in self.instance_dict:
                    raise ConflictError(f'duplicated instance key: {instance.key}')
                self.instance_dict[instance.key] = instance
            else:
                pass
        
        self.tool_mapping: Dict[str, ToolCollection] = {}
        self.tool_schemas: List[Dict] = []

        for tool in self.tools_list:
            self.tool_schemas.extend(tool.get_tools())
            names = tool.get_names()
            for name in names:
                if name in self.tool_mapping:
                    raise ConflictError(f'duplicated tool name: {name}')
                self.tool_mapping[name] = tool

    def __getitem__(self, instance_key: str) -> Any:
        return self.instance_dict.get(instance_key, None)

    async def build(self) -> None:
        await asyncio.gather(*call_coroutine_functions(
            self.tools_list + self.context_list,
            'build'
        ))
        return

    async def release(self) -> None:
        await asyncio.gather(*call_coroutine_functions(
            self.tools_list + self.context_list,
            'release'
        ))
        return
    
    async def reset(self) -> None:
        await asyncio.gather(*call_coroutine_functions(
            self.tools_list + self.context_list,
            'reset'
        ))
        return
    
    async def procedure(self, max_iteration: int = 30) -> None:
        async def tool_not_exists(original_call: ToolCall):
            return ToolReturn(
                content = f'[ERROR]: Tool {original_call.name} not exists. Please try again.',
                original_call = original_call
            )

        initial_prompt = ModelInput(messages = [], tools = self.tool_schemas)
        call_functions(self.context_list, 'process_prompt', model_input = initial_prompt)

        input: ModelInput = copy.deepcopy(initial_prompt)
        output: Optional[ModelResponse] = None
        for iter in range(max_iteration):
            print(input)
            output = await self.agent_model.generate(input)
            print(output)
            
            if output.tool_call is not None and len(output.tool_call) > 0:
                tool_call_tasks: List = []
                for tool_call in output.tool_call:
                    if tool_call.name in self.tool_mapping:
                        tool_call_tasks.append(self.tool_mapping[tool_call.name](tool_call))
                    else:
                        tool_call_tasks.append(tool_not_exists(tool_call))
                tool_returns = await asyncio.gather(*tool_call_tasks)
                input = self.agent_model.tool_return_integrate(input, output, tool_returns)
                continue

            if output.response != '':
                break

if __name__ == '__main__':
    from simplex.models.qwen import QwenConversationModel

    model = QwenConversationModel(
        base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key = 'sk-9b3a060c5d4d4c748af56ca372b9a9ed'
    )

    from simplex.tools.pyinterpreter import PythonInterpreter

    interpreter = PythonInterpreter(
        use_container = True,
        default_image = 'python:3.11-slim'
    )

    from simplex.context.base import InitPromptContext

    prompt = "**Solve the following, rounding your answer to three decimal places:** " \
             "In triangle \\( ABC \\), side \\( a = 7.5 \\), side \\( b = 9.2 \\), and \\( \angle C = 38.4^\\circ \\). " \
             "Find the length of side \\( c \\) using the Law of Cosines."

    init_prompt = InitPromptContext(
        system_prompt = 'You are a helpful assistant. Think briefly and answer the following question. Keep your thinking in no more than 10 sentences.',
        user_instruction = prompt
    )

    agent = AgentLoop(model, interpreter, init_prompt)

    async def test():
        await agent.build()
        await agent.procedure()
        await agent.release()

    asyncio.run(test())
