import os
import json
import uuid
import pathlib

from pathlib import Path
from enum import Enum, auto
from typing import Optional, List, Dict, Literal

import simplex.io
import simplex.basics
import simplex.tools.base

from simplex.io import UserInputInterface
from simplex.basics import (
    WebsocketClient,
    UnbuiltError,
    RequestError,
    UserNotify,
    UserResponse
)
from simplex.tools.base import (
    ToolCollection,
    ToolSchema,
    load_tool_definitions,
    load_schema
)


EditOperation = Literal[
    'view_workspace',
    'show_details',
    'view_file_content',
    'edit_file_content',
    'search',
    'create',
    'remove',
    'rename'
]

class EditTools(ToolCollection):
    SCHEMA_FILE: str = 'schema_edit_collections'

    def __init__(
        self,
        base_dir: str | Path,
        client: WebsocketClient,
        permission_required: bool = True,
        instance_id: Optional[str] = None, 
        rename_mapping: Dict[EditOperation, str] = {
            'view_workspace': 'view_workspace',
            'show_details': 'show_details',
            'view_file_content': 'view_file_content',
            'edit_file_content': 'edit_file_content',
            'search': 'search',
            'create': 'create',
            'remove': 'remove',
            'rename': 'rename'
        }
    ) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex, { value: f"_tool_{key}" for key, value in rename_mapping.items() })
        
        self.client = client
        self.permission_required = permission_required
        self.names = rename_mapping

        self.base_dir: Path = Path(base_dir)
        self.initialized: bool = False

        self.tool_definitions = load_tool_definitions(self.SCHEMA_FILE)
        self.view_workspace_schema = load_schema(self.SCHEMA_FILE, 'view_workspace', self.names['view_workspace'])
        self.show_details_schema = load_schema(self.SCHEMA_FILE, 'show_details', self.names['show_details'])
        self.view_file_content_schema = load_schema(self.SCHEMA_FILE, 'view_file_content', self.names['view_file_content'])
        self.edit_file_content_schema = load_schema(self.SCHEMA_FILE, 'edit_file_content', self.names['edit_file_content'])
        self.search_schema = load_schema(self.SCHEMA_FILE, 'search', self.names['search'])
        self.create_schema = load_schema(self.SCHEMA_FILE, 'create', self.names['create'])
        self.remove_schema = load_schema(self.SCHEMA_FILE, 'remove', self.names['remove'])
        self.rename_schema = load_schema(self.SCHEMA_FILE, 'rename', self.names['rename'])

        self.all_schemas: Dict[EditOperation, ToolSchema] = {
            'view_workspace': self.view_workspace_schema,
            'show_details': self.show_details_schema,
            'view_file_content': self.view_file_content_schema,
            'edit_file_content': self.edit_file_content_schema,
            'search': self.search_schema,
            'create': self.create_schema,
            'remove': self.remove_schema,
            'rename': self.rename_schema
        }

        self.schemas: Dict[EditOperation, ToolSchema] = { key: value for key, value in self.all_schemas.items() if key in self.names }

        self.input_interface: Optional[UserInputInterface] = None

    async def build(self) -> None:
        try:
            await self.client.build()
            self.initialized = True

            query: Dict = {
                'type': 'set_working_dir',
                'base_dir': str(self.base_dir.absolute())
            }
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
        except Exception:
            raise

    async def release(self) -> None:
        try:
            await self.client.release()
            self.initialized = False
        except Exception:
            raise

    async def reset(self) -> None:
        pass

    async def bind_io(self, input_interface: UserInputInterface, **kwargs) -> None:
        self.input_interface = input_interface

    def get_tool_schemas(self) -> List[ToolSchema]:
        return list(self.schemas.values())
    
    def tools_descriptions(self) -> str:
        return self.tool_definitions

    async def _tool_view_workspace(self, **kwargs) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        query: Dict = {
            'type': 'get_workspace_view'
        }

        try:
            response: Optional[str] = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise
    
    async def _tool_show_details(self, target_path: str, **kwargs) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        query: Dict = {
            'type': 'show_details',
            'target_path': target_path
        }

        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise
    
    async def _tool_view_file_content(
        self, 
        target_path: str, 
        line_start: Optional[int] = None, 
        line_end: Optional[int] = None, 
        **kwargs
    ) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        query: Dict = {
            'type': 'view_file_content',
            'target_path': target_path,
        }

        if line_start is not None:
            query['line_start'] = line_start
        if line_end is not None:
            query['line_end'] = line_end

        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise
    
    async def _tool_edit_file_content(
        self,
        target_path: str,
        edit_type: str,
        content: str,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
        **kwargs
    ) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        query: Dict = {
            'type': 'edit_file_content',
            'target_path': target_path,
            'edit_type': edit_type,
            'content': content
        }

        if line_start is not None:
            query['line_start'] = line_start
        if line_end is not None:
            query['line_end'] = line_end

        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise
    
    async def _tool_search(self, key_words: str, scope: str, mode: str, **kwargs) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        query: Dict = {
            'type': 'search_entity',
            'key_words': [word.strip() for word in key_words.split(',')],
            'scope': scope,
            'mode': mode
        }
        
        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise
    
    async def _tool_create(self, target_path: str, content: str, **kwargs) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        if self.input_interface and self.permission_required:
            user_response = await self.input_interface.notify_user(UserNotify('permission', f"Do you allow agent to create file: {target_path}?"))
            if not user_response.permitted:
                return f"[ERROR]: Permission error! {user_response.reason}"
        
        query: Dict = {
            'type': 'touch',
            'target_path': target_path,
            'content': content
        }
        
        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise
    
    async def _tool_remove(self, target_path: str, **kwargs) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        if self.input_interface and self.permission_required:
            user_response = await self.input_interface.notify_user(UserNotify('permission', f"Do you allow agent to remove file or directories: {target_path}?"))
            if not user_response.permitted:
                return f"[ERROR]: Permission error! {user_response.reason}"
        
        query: Dict = {
            'type': 'remove',
            'target_path': target_path
        }

        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise

    async def _tool_rename(self, src_path: str, dst_path: str, **kwargs) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)

        if self.input_interface and self.permission_required:
            user_response = await self.input_interface.notify_user(UserNotify('permission', f"Do you allow agent to move {src_path} to {dst_path}?"))
            if not user_response.permitted:
                return f"[ERROR]: Permission error! {user_response.reason}"
        
        query: Dict = {
            'type': 'rename',
            'src_path': src_path,
            'dst_path': dst_path
        }

        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise

if __name__ == '__main__':
    pass
