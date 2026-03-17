import os
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

from simplex.basics import ModelResponse, ToolCall
from simplex.models import MockConversationModel
from simplex.context import TrajectoryLogContext
from simplex.loop import AgentLoop, UserLoop
from simplex.tools import MockCalculator
from simplex.io import RichTerminalInterface


MODULE_PATH: Path = Path(__file__).resolve().parent
OUTPUT_PATH: Path = MODULE_PATH / 'output/test_interactive'

if __name__ == '__main__':
    async def test_body() -> None:
        model = MockConversationModel(
            expected_responses = [
                ModelResponse(tool_call = [ToolCall(uuid.uuid4().hex, 'calculator', {'operation': '+', 'operand1': 1, 'operand2': 1})]),
                ModelResponse(tool_call = [
                    ToolCall(uuid.uuid4().hex, 'calculator', {'operation': '/', 'operand1': 1, 'operand2': 0}),
                    ToolCall(uuid.uuid4().hex, 'calculator', {'random_params': None})
                ]),
                ModelResponse(response = 'The answer is 2.')
            ]
        )

        interface = RichTerminalInterface('cool agent')
        loop = AgentLoop(model, interface.get_exception_handler(), TrajectoryLogContext(instance_id = 'log'), MockCalculator())

        await UserLoop(interface, interface, loop).serve()

        target_path = OUTPUT_PATH / 'test_interactive.md'
        target_path.parent.mkdir(parents = True, exist_ok = True)
        with open(target_path, 'w', encoding = 'utf8') as file:
            file.write(loop['log'].human_readable) # type: ignore
    
    try:
        asyncio.run(test_body())
    except Exception:
        raise