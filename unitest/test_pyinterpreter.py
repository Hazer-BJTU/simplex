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

from simplex.basics import ContainerManager, PromptTemplate
from simplex.context import TrajectoryLogContext
from simplex.models import QwenConversationModel
from simplex.tools import PythonInterpreter
from simplex.loop import AgentLoop, LogExceptionHandler


PROBLEM_STATEMENT: str = """Please solve the following promblem.
#### Problem Statement
A small retail store tracks the daily sales of 5 different products over a week (7 days). The sales data (in units sold) is given as follows:

| Product/Day | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday |
|-------------|--------|---------|-----------|----------|--------|----------|--------|
| Product A   | 12     | 15      | 10        | 18       | 20     | 25       | 17     |
| Product B   | 8      | 10      | 9         | 11       | 13     | 15       | 12     |
| Product C   | 20     | 22      | 18        | 25       | 28     | 30       | 24     |
| Product D   | 5      | 7       | 6         | 8        | 9      | 10       | 8      |
| Product E   | 15     | 18      | 14        | 21       | 22     | 26       | 19     |

Using Python and NumPy, solve the following tasks:
1. Create a 2D NumPy array to represent the sales data (rows = products, columns = days).
2. Calculate the **total weekly sales** for each product (sum of sales across 7 days for each row).
3. Calculate the **average daily sales** for the entire store (average of all values in the array).
4. Find the **maximum number of units sold** for any product on a single day, and identify which product (A/B/C/D/E) and day (Monday-Sunday) it corresponds to.

#### Requirements
- Write a complete Python script using NumPy to solve all tasks.
- Print all results clearly (label each output).
- The final results must be verifiable with manual calculation.
"""
DEFAULT_IMAGE: str = 'playground-v1'
MODULE_PATH: Path = Path(__file__).resolve().parent
OUTPUT_PATH: Path = MODULE_PATH / 'output/test_pyinterpreter'

@pytest.mark.model_api_required
def test_pyinterpreter_qwen() -> None:
    async def test_body() -> None:
        model = QwenConversationModel(
            base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key = os.getenv('DASHSCOPE_API_KEY'), #type: ignore
            qwen_model = 'qwen3-coder-plus',
            enable_thinking = False
        )

        container = ContainerManager(name = 'test_pyinterpreter_qwen', default_image = DEFAULT_IMAGE)

        interpreter = PythonInterpreter(
            use_container = True, 
            container_manager = container,
            exec_command = lambda script: ['python', '-c', script]
        )

        async with AgentLoop(
            model, 
            LogExceptionHandler(),
            TrajectoryLogContext(instance_id = 'log'),
            interpreter
        ) as loop:
            result = await loop.complete(
                PromptTemplate('You are a helpful assistant.'),
                PromptTemplate(PROBLEM_STATEMENT)
            )
            log_content = loop['log'].human_readable # type: ignore
            target_path = OUTPUT_PATH / 'test_pyinterpreter_qwen.md'
            target_path.parent.mkdir(parents = True, exist_ok = True)
            with open(target_path, 'w', encoding = 'utf8') as file:
                file.write(log_content)

    try:
        asyncio.run(test_body())
    except Exception:
        raise

if __name__ == '__main__':
    pass
