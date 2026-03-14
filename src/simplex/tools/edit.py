import os
import json
import uuid
import pathlib

from pathlib import Path
from enum import Enum, auto
from typing import Optional, List, Dict

import simplex.basics
import simplex.tools.base

from simplex.basics import (
    WebsocketClient,
    UnbuiltError,
    RequestError
)
from simplex.tools.base import (
    ToolCollection,
    ToolSchema,
    load_tool_definitions,
    load_schema
)

class EditTools(ToolCollection):
    class Operation(Enum):
        view_workspace = auto()
        show_details = auto()
        view_file_content = auto()
        edit_file_content = auto()
        search = auto()
        create = auto()
        remove = auto()

    SCHEMA_FILE: str = 'schema_edit_collections'
    VIEW_WORKSPACE = Operation.view_workspace.name
    SHOW_DETAILS = Operation.show_details.name
    VIEW_FILE_CONTENT = Operation.view_file_content.name
    EDIT_FILE_CONTENT = Operation.edit_file_content.name
    SEARCH = Operation.search.name
    CREATE = Operation.create.name
    REMOVE = Operation.remove.name

    def __init__(
        self,
        base_dir: str | Path,
        client: WebsocketClient,
        instance_id: Optional[str] = None, 
        rename_mapping: Dict[str, str] = {
            VIEW_WORKSPACE: 'view_workspace',
            SHOW_DETAILS: 'show_details',
            VIEW_FILE_CONTENT: 'view_file_content',
            EDIT_FILE_CONTENT: 'edit_file_content',
            SEARCH: 'search',
            CREATE: 'create',
            REMOVE: 'remove'
        }
    ) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex, { value: f"_tool_{key}" for key, value in rename_mapping.items() })
        
        self.client = client
        self.names = rename_mapping

        self.base_dir: Path = Path(base_dir)
        self.initialized: bool = False

        self.tool_definitions = load_tool_definitions(self.SCHEMA_FILE)
        self.view_workspace_schema = load_schema(self.SCHEMA_FILE, self.VIEW_WORKSPACE, self.names[self.VIEW_WORKSPACE])
        self.show_details_schema = load_schema(self.SCHEMA_FILE, self.SHOW_DETAILS, self.names[self.SHOW_DETAILS])
        self.view_file_content_schema = load_schema(self.SCHEMA_FILE, self.VIEW_FILE_CONTENT, self.names[self.VIEW_FILE_CONTENT])
        self.edit_file_content_schema = load_schema(self.SCHEMA_FILE, self.EDIT_FILE_CONTENT, self.names[self.EDIT_FILE_CONTENT])
        self.search_schema = load_schema(self.SCHEMA_FILE, self.SEARCH, self.names[self.SEARCH])
        self.create_schema = load_schema(self.SCHEMA_FILE, self.CREATE, self.names[self.CREATE])
        self.remove_schema = load_schema(self.SCHEMA_FILE, self.REMOVE, self.names[self.REMOVE])

        self.schemas: Dict[str, ToolSchema] = {
            self.VIEW_FILE_CONTENT: self.view_file_content_schema,
            self.SHOW_DETAILS: self.show_details_schema,
            self.VIEW_FILE_CONTENT: self.view_file_content_schema,
            self.EDIT_FILE_CONTENT: self.edit_file_content_schema,
            self.SEARCH: self.search_schema,
            self.CREATE: self.create_schema,
            self.REMOVE: self.remove_schema
        }

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

if __name__ == '__main__':
    pass
