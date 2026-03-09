import os
import json
import uuid
import pytest
import asyncio

import simplex.basics
import simplex.context
import simplex.models
import simplex.tools
import simplex.loop

from simplex.basics import ModelInput, ModelResponse, ToolCall
from simplex.context import TrajectoryLogContext
from simplex.models import MockConversationModel
from simplex.tools import MockCalculator
from simplex.loop import AgentLoop


@pytest.mark.general_logic
def test_mock_loop() -> None:
    async def test_body() -> None:
        model = MockConversationModel(
            expected_responses = [
                ModelResponse(tool_call = [ToolCall(uuid.uuid4().hex, 'calculator', {'operation': '+', 'operand1': 1, 'operand2': 1})]),
                ModelResponse(response = 'The answer is two.')
            ]
        )

        async with AgentLoop(model, TrajectoryLogContext(instance_id = 'log'), MockCalculator()) as loop:
            await loop.procedure()
            log_content = loop['log'].get()

        print(json.dumps(log_content, indent = 2))
        
    asyncio.run(test_body())

if __name__ == '__main__':
    pass
