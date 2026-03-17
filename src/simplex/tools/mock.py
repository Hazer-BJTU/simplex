import os
import uuid
import asyncio

from typing import List, Optional

import simplex.io
import simplex.basics
import simplex.tools.base

from simplex.basics import (
    ModelInput,
    ContainerManager,
    UnbuiltError,
    EntityInitializationError,
    ParameterError,
    UserNotify,
    UserResponse
)
from simplex.tools.base import (
    ToolSchema,
    ToolCollection,
    load_schema,
    load_tool_definitions
)
from simplex.io import UserInputInterface

class MockCalculator(ToolCollection):
    SCHEMA_FILE: str = 'schema_mock'
    CALCULATOR: str = 'calculator'

    def __init__(
        self, 
        instance_id: Optional[str] = None, 
        rename: str = 'calculator',
        ask_for_permission: bool = True
    ) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex, { rename: '_tool_calculator' })

        self.name = rename
        self.ask_for_permission = ask_for_permission

        self.tool_definition = load_tool_definitions(self.SCHEMA_FILE)
        self.schema = load_schema(self.SCHEMA_FILE, self.CALCULATOR, self.name)

        self.input_interface: Optional[UserInputInterface] = None

    async def build(self) -> None:
        pass

    async def release(self) -> None:
        pass

    async def reset(self) -> None:
        pass

    def clone(self) -> ToolCollection:
        return MockCalculator(
            uuid.uuid4().hex,
            self.name,
            self.ask_for_permission
        )
    
    async def bind_io(self, input_interface: UserInputInterface, **kwargs) -> None:
        self.input_interface = input_interface

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [self.schema]
    
    def tools_descriptions(self) -> str:
        return self.tool_definition

    async def _tool_calculator(self, operation: str, operand1: float, operand2: float, **kwargs) -> str:
        if self.ask_for_permission and self.input_interface:
            response = await self.input_interface.notify_user(UserNotify('permission', f"Do you allow model to use calculator to perform '{operation}'?"))
            if not response.permitted:
                return f"[ERROR]: Permission error! {response.reason}"

        if operation == 'add' or operation == '+':
            return str(operand1 + operand2)
        elif operation == 'subtract' or operation == '-':
            return str(operand1 - operand2)
        elif operation == 'multiply' or operation == '*':
            return str(operand1 * operand2)
        elif operation == 'divide' or operation == '/':
            if abs(operand2) < 1e-8:
                raise ZeroDivisionError('operand2 should not be zero when performing division')
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
