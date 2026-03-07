import os
import uuid
import time
import queue
import asyncio
import threading
import websockets

from datetime import datetime
from typing import Optional, Dict, Any
from websockets.sync.client import connect as sync_connect
from websockets.asyncio.client import connect as async_connect
from websockets.exceptions import ConnectionClosed

import simplex.basics.exception

from simplex.basics.exception import UnbuiltError


class WebsocketClient:
    def __init__(
        self, 
        port: int,
        host: str = 'localhost',
        max_queue_size: int = 0,
        max_retry: int = 5,
        await_timeout: float = 1
    ) -> None:
        self.port = port
        self.host = host
        self.max_queue_size = max_queue_size
        self.max_retry = max_retry
        self.await_timeout = await_timeout

        self.url: str = f'ws://{self.host}:{self.port}'
        self.data_queue: asyncio.Queue = asyncio.Queue(maxsize = self.max_queue_size)
        self.result_dict: Dict[str, asyncio.Future] = {}
        self.initialized: bool = False
        self.exit_flag: bool = False

        self.client_task: Optional[asyncio.Task] = None

    async def _clear_all(self) -> None:
        for futures in self.result_dict.values():
            try:
                futures.set_result(None)
            except Exception:
                pass
        self.result_dict.clear()
        self.data_queue = asyncio.Queue(maxsize = self.max_queue_size)

    async def _connection_io(self, websocket) -> None:
        while not self.exit_flag:
            try:
                id, data = await asyncio.wait_for(self.data_queue.get(), timeout = self.await_timeout)
                await websocket.send(data)
                result: Any = await websocket.recv()
                if id in self.result_dict:
                    self.result_dict[id].set_result(result)
            except asyncio.TimeoutError:
                continue
            except Exception:
                await self.data_queue.put((id, data)) #type: ignore
                raise

    async def _start_client(self, max_retry: Optional[int] = None) -> None:
        if max_retry is None:
            max_retry = self.max_retry

        try:
            retry_count: int = 0
            while retry_count < self.max_retry:
                try:
                    async with async_connect(self.url, open_timeout = self.await_timeout) as websocket:
                        retry_count = 0
                        await self._connection_io(websocket = websocket)
                        return
                except Exception:
                    retry_count += 1
                    await asyncio.sleep(0.5)
                    continue
            
            self.exit_flag = True
            await self._clear_all()
        except Exception:
            raise

    async def build(self) -> None:
        if self.initialized:
            return
        
        self.exit_flag = False
        self.client_task = asyncio.create_task(self._start_client())
        self.initialized = True

    async def release(self) -> None:
        self.exit_flag = True

        if self.client_task is not None:
            try:
                await self.client_task
            except Exception:
                raise
            
        await self._clear_all()
        self.initialized = False

    async def exchange(self, data: Any) -> Any:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        if self.exit_flag:
            return None

        try:
            data_id: str = uuid.uuid4().hex
            future: asyncio.Future = asyncio.Future()
            self.result_dict[data_id] = future
            await self.data_queue.put((data_id, data))
            result = await future
            if data_id in self.result_dict:
                self.result_dict.pop(data_id)
            return result
        except Exception:
            raise

    async def __aenter__(self):
        await self.build()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.release()
        return False
    
class WebsocketClientSync:
    def __init__(
        self,
        port: int,
        host: str = 'localhost',
        max_queue_size: int = 0,
        max_retry: int = 5,
        await_timeout: float = 1
    ) -> None:
        self.port = port
        self.host = host
        self.max_queue_size = max_queue_size
        self.max_retry = max_retry
        self.await_timeout = await_timeout

        self.url: str = f'ws://{self.host}:{self.port}'
        self.data_queue: queue.Queue = queue.Queue(maxsize = self.max_queue_size)
        self.event_dict: Dict[str, threading.Event] = {}
        self.result_dict: Dict[str, Any] = {}
        self.dict_mtx = threading.Lock()
        self.initialized: bool = False
        self.exit_flag: bool = False

        self.client_task: Optional[threading.Thread] = None

    def _clear_all(self) -> None:
        with self.dict_mtx:
            for event in self.event_dict.values():
                event.set()
            self.event_dict.clear()
            self.result_dict.clear()
            self.data_queue = queue.Queue(maxsize = self.max_queue_size)
    
    def _connection_io(self, websocket) -> None:
        while not self.exit_flag:
            try:
                id, data = self.data_queue.get(timeout = self.await_timeout)
                websocket.send(data)
                result: Any = websocket.recv()
                with self.dict_mtx:
                    if id in self.event_dict:
                        self.event_dict[id].set()
                        self.result_dict[id] = result
            except queue.Empty:
                continue
            except TimeoutError:
                continue
            except Exception:
                self.data_queue.put((id, data)) #type: ignore
                raise

    def _start_client(self, max_retry: Optional[int] = None) -> None:
        if max_retry is None:
            max_retry = self.max_retry

        try:
            retry_count: int = 0
            while retry_count < self.max_retry:
                try:
                    with sync_connect(self.url, open_timeout = self.await_timeout) as websocket:
                        retry_count = 0
                        self._connection_io(websocket)
                        return
                except Exception:
                    retry_count += 1
                    time.sleep(0.5)
                    continue

            self.exit_flag = True
            self._clear_all()
        except Exception:
            raise

    def build(self) -> None:
        if self.initialized:
            return
        
        self.exit_flag = False
        self.client_task = threading.Thread(target = self._start_client)
        self.client_task.start()
        self.initialized = True

    def release(self) -> None:
        self.exit_flag = True

        if self.client_task is not None:
            try:
                self.client_task.join()
            except Exception:
                raise

        self._clear_all()
        self.initialized = False

    def exchange(self, data: Any) -> Any:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        if self.exit_flag:
            return None
        
        try:
            data_id: str = uuid.uuid4().hex
            event: threading.Event = threading.Event()
            with self.dict_mtx:
                self.event_dict[data_id] = event
            self.data_queue.put((data_id, data))
            event.wait()
            with self.dict_mtx:
                result = self.result_dict.get(data_id, None)
                if data_id in self.event_dict:
                    self.event_dict.pop(data_id)
                if data_id in self.result_dict:
                    self.result_dict.pop(data_id)
            return result
        except Exception:
            raise

    def __enter__(self):
        self.build()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

if __name__ == '__main__':
    pass
    