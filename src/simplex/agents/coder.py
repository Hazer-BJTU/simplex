import os
import json
import pathlib

from pathlib import Path
from typing import Optional

import simplex.basics
import simplex.models
import simplex.loop
import simplex.context
import simplex.tools

from simplex.basics import (
    ExceptionHandler,
    LogExceptionHandler,
    WebsocketClient
)
from simplex.models import ConversationModel
from simplex.loop import AgentLoop, AgentLoopAdapter
from simplex.context import TrajectoryLogContext, TokenCostCounter
from simplex.tools import EditTools


def get_standard_coder_agent(
    model: ConversationModel,
    work_dir: str | Path,
    exception_handler: Optional[ExceptionHandler] = None,
    edit_tools_port: int = 9002,
    edit_tools_host: str = 'localhost',
    log: str = 'log',
    token_counter: str = 'token_counter'
) -> AgentLoopAdapter:
    return AgentLoop(
        model,
        exception_handler if exception_handler is not None else LogExceptionHandler(),
        EditTools(work_dir, WebsocketClient(edit_tools_port, edit_tools_host)),
        TrajectoryLogContext(instance_id = log),
        TokenCostCounter(instance_id = token_counter)
    )


if __name__ == '__main__':
    pass
