import os
import sys
import time
import asyncio

import simplex.basics
import simplex.tools

from simplex.basics import WebsocketClient, ToolCall, ToolReturn
from simplex.tools import EditTools


if __name__ == '__main__':
    async def test() -> None:
        async with EditTools('/home/hazer/simplex', WebsocketClient(9002)) as tool:
            start = time.time()
            ret: ToolReturn = await tool(ToolCall('1', 'search', {'scope': 'global', 'mode': 'pattern', 'key_words': 'GEMCLnetwork'}))
            end = time.time()
            print(ret.content)
            print(f"Execution time: {end - start:.4f}s")

            key = input()



    asyncio.run(test())