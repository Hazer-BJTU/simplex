import os
import json
import uuid
import pathlib
import asyncio

from pathlib import Path

import simplex.basics
import simplex.models
import simplex.context
import simplex.loop
import simplex.tools
import simplex.io

from simplex.basics import WebsocketClient, ModelResponse, ToolCall
from simplex.models import QwenConversationModel, MockConversationModel
from simplex.context import TrajectoryLogContext, TokenCostCounter, ActionSelfEvaluation
from simplex.loop import AgentLoop, UserLoop
from simplex.tools import EditTools, SubprocessExecutorLocal, SequentialPlan
from simplex.io import RichTerminalInterface


MODULE_PATH: Path = Path(__file__).resolve().parent
OUTPUT_PATH: Path = MODULE_PATH / 'output/test_interactive'

if __name__ == '__main__':
    async def test_body() -> None:
        model = QwenConversationModel('https://dashscope.aliyuncs.com/compatible-mode/v1', os.getenv('API_KEY'), qwen_model = 'glm-5') # type: ignore

        
        model_mock = MockConversationModel(
            expected_responses = [
                ModelResponse(tool_call = [ToolCall('x', 'make_plan', {'content': 'This is my plan A.', 'edit_type': 'append'})]),
                ModelResponse(response = 'Hello!'),
                ModelResponse(tool_call = [ToolCall('x', 'make_plan', {'content': 'This is my plan B.', 'edit_type': 'append'})]),
                ModelResponse(tool_call = [ToolCall('x', 'make_plan', {'content': 'This is my plan B.', 'edit_type': 'replace'})]),
                ModelResponse(tool_call = [ToolCall('x', 'make_plan', {'content': '', 'edit_type': 'check_only'})]),
                ModelResponse(response = 'Hello!')
            ]
        )

        interface = RichTerminalInterface(model.qwen_model)
        loop = AgentLoop(
            model, 
            interface.get_exception_handler(), 
            TrajectoryLogContext(instance_id = 'log'), 
            EditTools('/home/hazer/simplex', WebsocketClient(9002)),
            SubprocessExecutorLocal(),
            SequentialPlan(),
            TokenCostCounter(),
            ActionSelfEvaluation()
        )

        await UserLoop(interface, interface, loop, complete_configs = {'max_iteration': 100}).serve()

        target_path = OUTPUT_PATH / 'test_interactive.md'
        target_path.parent.mkdir(parents = True, exist_ok = True)
        with open(target_path, 'w', encoding = 'utf8') as file:
            file.write(loop['log'].human_readable) # type: ignore

        target_path = OUTPUT_PATH / 'test_ineractive.json'
        target_path.parent.mkdir(parents = True, exist_ok = True)
        with open(target_path, 'w', encoding = 'utf8') as file:
            file.write(json.dumps(loop['log'].dictionary, indent = 2)) #type: ignore

    try:
        asyncio.run(test_body())
    except Exception:
        raise