import os
import json
import asyncio

import simplex.basics.client
import simplex.loop.base
import simplex.context.base
import simplex.tools.edit
import simplex.models.qwen

from simplex.basics.client import WebsocketClient
from simplex.loop.base import AgentLoop
from simplex.context.base import TrajectoryLogContext, InitPromptContext
from simplex.tools.edit import EditTools
from simplex.models.qwen import QwenConversationModel


if __name__ == '__main__':
    prompt = InitPromptContext(
        system_prompt = 'You are a skillful developer.',
        user_instruction = 'Please help me to write detailed annotations for src/simplex/models/qwen.py based on your understanding. ' \
                           'In your final response, briefly summarize the changes that you have made to the project. ' \
                           'I will review your code and commit the changes to repository. '
    )
    tools = EditTools(WebsocketClient(9002, 'localhost'))
    model = QwenConversationModel(
        base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key = os.getenv('DASHSCOPE_API_KEY'), #type: ignore
        qwen_model = 'qwen3-coder-plus',
        enable_thinking = False
    )
    logger = TrajectoryLogContext('log')

    loop = AgentLoop(model, prompt, logger, tools)

    async def test() -> None:
        await loop.build()
        await loop.procedure(max_iteration = 50)
        await loop.release()

        output = loop['log'].get()
        with open('trajectory.jsonl', 'w', encoding = 'utf8') as file:
            file.write(json.dumps(output, indent = 2))

    asyncio.run(test())
    # print(tools.get_tools())
    # print(tools.get_names())