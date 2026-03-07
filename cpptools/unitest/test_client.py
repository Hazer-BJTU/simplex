import os
import json
import asyncio

from simplex.basics.client import WebsocketClient, WebsocketClientSync

if __name__ == "__main__":
    with WebsocketClientSync(9002, 'localhost') as client:
        print(client.exchange(json.dumps({'type': 'set_working_dir', 'base_dir': '/home/hazer/simplex/'})))
        print(client.exchange(json.dumps({
            'type': 'touch', 
            'target_path': 'cpptools/bin/tests/output.txt',
            'content': 'Hello world#1!\n    Hello world#2!\n        Hello world#3!\n    Hello world#4!\nHello world#5!\n'
        })))
        print(client.exchange(json.dumps({
            'type': 'remove',
            'target_path': 'cpptools/bin/tests/output.txt'
        })))
    