import os
import json
import uuid
import pathlib
import pytest
import asyncio

from pathlib import Path

import simplex.basics
import simplex.context
import simplex.models
import simplex.tools
import simplex.loop
import simplex.io

from simplex.basics import PromptTemplate, ModelResponse, ToolCall
from simplex.context import TrajectoryLogContext
from simplex.models import MockConversationModel
from simplex.tools import MockCalculator
from simplex.loop import AgentLoop, LogExceptionHandler, UserLoop
from simplex.io import RichTerminalInterface


MODULE_PATH: Path = Path(__file__).resolve().parent
OUTPUT_PATH: Path = MODULE_PATH / 'output/test_logic'

@pytest.mark.no_requirements
def test_mock_loop() -> None:
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

        async with AgentLoop(model, LogExceptionHandler(), TrajectoryLogContext(instance_id = 'log'), MockCalculator()) as loop:
            response = await loop.complete(
                system = PromptTemplate('You are a helpful assistant'),
                user = PromptTemplate('This is a test.')
            )
            detailed_log = loop['log'].dictionary # type: ignore
            markdown_log = loop['log'].human_readable # type: ignore

        target_path = OUTPUT_PATH / 'test_mock_loop_markdown.md'
        target_path.parent.mkdir(parents = True, exist_ok = True)
        with open(target_path, 'w', encoding = 'utf8') as file:
            file.write(markdown_log)

        target_path = OUTPUT_PATH / 'test_mock_loop_detailed.txt'
        target_path.parent.mkdir(parents = True, exist_ok = True)
        with open(target_path, 'w', encoding = 'utf8') as file:
            file.write(json.dumps(detailed_log, indent = 2))

        # print(response)
    
    try:
        asyncio.run(test_body())
    except Exception:
        raise

@pytest.mark.no_requirements
def test_mock_user_loop() -> None:
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

        loop = AgentLoop(model, LogExceptionHandler(), TrajectoryLogContext(instance_id = 'log'), MockCalculator())

        interface = RichTerminalInterface('cool agent')

        await UserLoop(interface, interface, loop).serve()

        target_path = OUTPUT_PATH / 'test_mock_user_loop_markdown.md'
        target_path.parent.mkdir(parents = True, exist_ok = True)
        with open(target_path, 'w', encoding = 'utf8') as file:
            file.write(loop['log'].human_readable) # type: ignore
    
    try:
        asyncio.run(test_body())
    except Exception:
        raise

if __name__ == '__main__':
    pass
