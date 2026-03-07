import os
import json
import asyncio

import simplex.basics.dataclass
import simplex.basics.client
import simplex.tools.edit

from simplex.basics.dataclass import ToolCall, ToolReturn
from simplex.basics.client import WebsocketClient
from simplex.tools.edit import EditTools


if __name__ == '__main__':
    async def test() -> None:
        async with EditTools(WebsocketClient(9002, 'localhost')) as tools:
            test1: ToolCall = ToolCall('#1', 'view_workspace', {})
            output1: ToolReturn = await tools(test1)
            print(output1.content)
            print()

            test2: ToolCall = ToolCall('#2', 'show_details', {'target_path': 'cpptools'})
            output2: ToolReturn = await tools(test2)
            print(output2.content)
            print()

            test3: ToolCall = ToolCall('#3', 'show_details', {'target_path': 'cpptools/CMakeLists.txt'})
            output3: ToolReturn = await tools(test3)
            print(output3.content)
            print()

            test4: ToolCall = ToolCall('#4', 'show_details', {'target_path': 'cpptools/not_exists'})
            output4: ToolReturn = await tools(test4)
            print(output4.content)
            print()

            test5: ToolCall = ToolCall('#5', 'show_details', {'target_path': 'src/simplex/basics/client.py'})
            output5: ToolReturn = await tools(test5)
            print(output5.content)
            print()

            test6: ToolCall = ToolCall('#6', 'view_file_content', {'target_path': 'src/simplex/basics/client.py'})
            output6: ToolReturn = await tools(test6)
            print(output6.content)
            print()

            test7: ToolCall = ToolCall('#7', 'view_file_content', {'target_path': 'src/simplex/basics/client.py', 'line_start': 21, 'line_end': 127})
            output7: ToolReturn = await tools(test7)
            print(output7.content)
            print()

            test8: ToolCall = ToolCall('#8', 'create_file', {'target_path': 'unitest/hello.txt', 'content': 'Hello world!\n\tHello world!\nHello world!\n'})
            output8: ToolReturn = await tools(test8)
            print(output8.content)
            print()

            test9: ToolCall = ToolCall('#9', 'edit_file_content', {'target_path': 'unitest/hello.txt', 'edit_type': 'insert', 'line_start': 2, 'content': '\tThis is an additional line!\n'})
            output9: ToolReturn = await tools(test9)
            print(output9.content)
            print()

            test10: ToolCall = ToolCall('#10', 'edit_file_content', {'target_path': 'unitest/hello.txt', 'edit_type': 'insert', 'line_start': 10, 'content': 'This is also an additional line!\n'})
            output10: ToolReturn = await tools(test10)
            print(output10.content)
            print()

            test11: ToolCall = ToolCall('#11', 'edit_file_content', {'target_path': 'unitest/hello.txt', 'edit_type': 'insert', 'line_start': -1, 'content': 'QAQ\n'})
            output11: ToolReturn = await tools(test11)
            print(output11.content)
            print()

            test12: ToolCall = ToolCall('#12', 'search', {'key_words': 'ContainerManager, on_init_output, _tool_view_workspace', 'scope': 'global', 'mode': 'semantic_search'})
            output12: ToolReturn = await tools(test12)
            print(output12.content)
            print()

            test13: ToolCall = ToolCall('#13', 'search', {'key_words': 'abstractmethod', 'scope': 'global', 'mode': 'pattern_match'})
            output13: ToolReturn = await tools(test13)
            print(output13.content)
            print()

            testf: ToolCall = ToolCall('#f', 'remove_file', {'target_path': 'unitest/hello.txt'})
            outputf: ToolReturn = await tools(testf)
            print(outputf.content)
            print()

    asyncio.run(test())
