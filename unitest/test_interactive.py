import os
import json
import uuid
import pickle
import pathlib
import asyncio

from pathlib import Path

import simplex.basics
import simplex.models
import simplex.context
import simplex.loop
import simplex.tools
import simplex.io

from simplex.basics import WebsocketClient, ToolCall, CommandProcess, ModelResponse
from simplex.models import MockConversationModel
from simplex.context import TrajectoryLogContext, TokenCostCounter, RollContextClipper
from simplex.loop import AgentLoop, UserLoop
from simplex.tools import EditTools, SubprocessExecutorLocal, SequentialPlan, InLoopConversation
from simplex.io import RichTerminalInterface


MODULE_PATH: Path = Path(__file__).resolve().parent
OUTPUT_PATH: Path = MODULE_PATH / 'output/test_interactive'

if __name__ == '__main__':
    async def test_body() -> None:
        model_mock = MockConversationModel(
            expected_responses = [
                ModelResponse(tool_call = [ToolCall('#1', 'operate_filesystem', {'operation': 'create', 'target_path': 'test.txt', 'content': 'Hello world!'})]),
                ModelResponse(tool_call = [ToolCall('#2', 'operate_filesystem', {'operation': 'remove', 'target_path': 'test.txt'})]),
                ModelResponse(tool_call = [ToolCall('#3', 'propose', {'content': 'Do you think the answer is 42?'})]),
                ModelResponse(response = 'The answer is 42.')
            ]
        )

        interface = RichTerminalInterface('cool agent')
        loop = AgentLoop(
            model_mock, 
            interface.get_exception_handler(), 
            TrajectoryLogContext(instance_id = 'log'), 
            EditTools('/home/hazer/simplex/examples/workspace', WebsocketClient(9002)),
            SubprocessExecutorLocal(),
            SequentialPlan(),
            RollContextClipper(),
            TokenCostCounter(),
            InLoopConversation()
        )

        await UserLoop(interface, interface, loop, complete_configs = {'max_iteration': 100}).serve()

        target_path = OUTPUT_PATH / 'test_interactive.md'
        target_path.parent.mkdir(parents = True, exist_ok = True)
        with open(target_path, 'w', encoding = 'utf8') as file:
            file.write(loop['log'].human_readable) # type: ignore

        target_path = OUTPUT_PATH / 'test_ineractive.pkl'
        target_path.parent.mkdir(parents = True, exist_ok = True)
        with open(target_path, 'wb') as file:
            pickle.dump(loop['log'].dictionary, file) # type: ignore

    try:
        with CommandProcess("simplex_tool_server -p 9002") as proc:
            asyncio.run(test_body())
    except Exception:
        raise