import os
import pytest
import difflib
import pathlib
import asyncio

from pathlib import Path
from difflib import SequenceMatcher

from simplex.basics import ToolCall, ToolReturn, WebsocketClient
from simplex.tools import EditTools


PORT: int = 9002
HOST: str = 'localhost'
MODULE_PATH: Path = Path(__file__).resolve().parent
MOCK_PROJ_PATH: Path = MODULE_PATH / 'fixtures/python_example/GeCoSleep'

@pytest.mark.tool_server_required
def test_client_async() -> None:
    async def test_body() -> None:
        output: str = ''
        async with EditTools(MOCK_PROJ_PATH, WebsocketClient(PORT, HOST)) as tools:
            async def sub_test(*args, **kwargs) -> None:
                nonlocal output
                input: ToolCall = ToolCall(*args, **kwargs)
                response: ToolReturn = await tools(input)
                output += response.content + '\n\n'
            await sub_test('#1', 'view_workspace', {})
            await sub_test('#2', 'show_details', {'target_path': 'BayesEEGNet'})
            await sub_test('#3', 'show_details', {'target_path': 'path_not_exists'})
            await sub_test('#4', 'show_details', {'target_path': 'README.md'})
            await sub_test('#5', 'show_details', {'target_path': 'GeCoSleep/EEGGR.py'})
            await sub_test('#6', 'view_file_content', {'target_path': 'GeCoSleep/generator.py'})
            await sub_test('#7', 'view_file_content', {'target_path': 'path_not_exists'})
            await sub_test('#8', 'view_file_content', {'target_path': 'GeCoSleep/EEGGR.py', 'line_start': 34, 'line_end': 177})
            await sub_test('#9', 'view_file_content', {'target_path': 'GeCoSleep/EEGGR.py', 'line_start': 117, 'line_end': 512})
            await sub_test('#10', 'view_file_content', {'target_path': 'GeCoSleep/EEGGR.py', 'line_start': -1, 'line_end': 20})
            await sub_test('#11', 'view_file_content', {'target_path': 'GeCoSleep/EEGGR.py', 'line_end': 20})
            await sub_test('#12', 'create', {'target_path': 'edit_hello.txt', 'content': 'Hello world!\n\tHello world!\nHello world!\n'})
            await sub_test('#13', 'create', {'target_path': 'edit_hello.txt', 'content': 'Hello world!\n\tHello world!\nHello world!\n\tHello world!\nHello world!\n'})
            await sub_test('#14', 'edit_file_content', {'target_path': 'edit_hello.txt', 'edit_type': 'insert', 'line_start': 2, 'content': 'This is an additional row!'})
            await sub_test('#15', 'edit_file_content', {'target_path': 'edit_hello.txt', 'edit_type': 'insert', 'line_start': 20, 'content': 'This is a additional row!'})
            await sub_test('#16', 'edit_file_content', {'target_path': 'edit_hello.txt', 'edit_type': 'replace', 'line_start': 1, 'line_end': 3, 'content': 'The rows are replaced!\nThe rows are replaced!'})
            await sub_test('#17', 'edit_file_content', {'target_path': 'edit_hello.txt', 'edit_type': 'replace', 'line_start': -1, 'line_end': 10, 'content': ''})
            await sub_test('#18', 'edit_file_content', {'target_path': 'edit_hello.txt', 'edit_type': 'replace', 'content': ''})
            await sub_test('#19', 'remove', {'target_path': 'edit_hello.txt'})
            await sub_test('#20', 'search', {'key_words': 'logdocument, MultiScaleCNN, load_data_isruc1', 'scope': 'global', 'mode': 'semantic_search'})
            await sub_test('#21', 'search', {'key_words': 'optimizer', 'scope': 'global', 'mode': 'pattern_match'})
        print(output)

    try:
        output = asyncio.run(test_body())
    except Exception:
        raise

if __name__ == '__main__':
    pass
