import os
import asyncio

import simplex.basics
import simplex.models
import simplex.context
import simplex.loop
import simplex.tools
import simplex.io

from simplex.basics import WebsocketClient
from simplex.models import   DeepSeekConversationModel
from simplex.context import TrajectoryLogContext, TokenCostCounter, RollContextClipper
from simplex.loop import AgentLoop, UserLoop
from simplex.tools import EditTools, SubprocessExecutorLocal, SequentialPlan
from simplex.io import RichTerminalInterface


if __name__ == '__main__':
    async def test_body() -> None:
        model = DeepSeekConversationModel('https://api.deepseek.com/beta', os.getenv('API_KEY'), model = 'deepseek-reasoner') # type: ignore

        interface = RichTerminalInterface(model.model)
        loop = AgentLoop(
            model, 
            interface.get_exception_handler(), 
            TrajectoryLogContext(instance_id = 'log'), 
            EditTools('/home/hazer/simplex', WebsocketClient(9002)),
            SubprocessExecutorLocal(),
            SequentialPlan(),
            RollContextClipper(threshold_ratio = 0.3, keep_fc_msgs = 30),
            TokenCostCounter()
        )

        await UserLoop(interface, interface, loop, complete_configs = {'max_iteration': 100}).serve()

    try:
        asyncio.run(test_body())
    except Exception:
        raise