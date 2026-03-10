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

from simplex.basics import ModelInput, ModelResponse, ToolCall
from simplex.context import TrajectoryLogContext, InitPromptContext
from simplex.models import MockConversationModel
from simplex.tools import MockCalculator
from simplex.loop import AgentLoop


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

        async with AgentLoop(model, InitPromptContext('This is a test.'), TrajectoryLogContext(instance_id = 'log'), MockCalculator()) as loop:
            await loop.procedure()
            log_content = loop['log'].human_readable

        target_path = OUTPUT_PATH / 'test_mock_loop.md'
        target_path.parent.mkdir(parents = True, exist_ok = True)
        with open(target_path, 'w', encoding = 'utf8') as file:
            file.write(log_content)
    
    try:
        asyncio.run(test_body())
    except Exception:
        raise

if __name__ == '__main__':
    pass
