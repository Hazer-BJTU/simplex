import os
import uuid
import asyncio

from typing import Optional, List, Dict, Callable, TYPE_CHECKING

import simplex.basics
import simplex.tools.base

from simplex.basics import (
    ModelInput,
    ContainerManager,
    UnbuiltError,
    EntityInitializationError,
    ParameterError
)
from simplex.tools.base import (
    ToolSchema,
    ToolCollection,
    load_schema,
    load_tool_definitions
)

if TYPE_CHECKING:
    import simplex.loop

    from simplex.loop import AgentLoop


class MockCalculator(ToolCollection):
    SCHEMA_FILE: str = 'schema_mock_calculator'
    CALCULATOR: str = 'calculator'

    def __init__(
        self, 
        instance_id: str, 
        rename: str = 'calculator'
    ) -> None:
        super().__init__(instance_id, { rename: '_tool_calculator' })

        self.name = rename

        self.tool_definition = load_tool_definitions(self.SCHEMA_FILE)
        self.schema = load_schema(self.SCHEMA_FILE, self.CALCULATOR)

    async def build(self) -> None:
        pass

    async def release(self) -> None:
        pass

    async def reset(self) -> None:
        pass

    def get_names(self) -> List[str]:
        return [self.name]
    
    def get_tools(self) -> List[ToolSchema]:
        return [self.schema]
    
    def tools_descriptions(self) -> str:
        return self.tool_definition
    
    def on_init_output(self, model_input: ModelInput, agent: simplex.tools.base.AgentLoop) -> None:
        pass

    def _tool_calculator(self, operation: str, operand1: float, operand2: float) -> str:
        if operation == 'add' or operation == '+':
            return str(operand1 + operand2)
        elif operation == 'subtract' or operation == '-':
            return str(operand1 - operand2)
        elif operation == 'multiply' or operation == '*':
            return str(operand1 * operand2)
        elif operation == 'divide' or operation == '/':
            if abs(operand2) < 1e-8:
                raise ZeroDivisionError
            return str(operand1 / operand2)
        else:
            raise ParameterError(
                '_tool_calculator',
                'operation',
                f"unknown: {operation}",
                'string',
                self.__class__.__name__
            )

if __name__ == '__main__':
    pass
