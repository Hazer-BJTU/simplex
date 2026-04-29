import os
import re
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
    UserResponse,
    AgentLoopStateEdit,
    PromptTemplate
)
from simplex.tools.base import (
    ToolCollection,
    ToolSchema,
    load_tool_definitions,
    load_schema,
    load_tool_skill
)


EditOperation = Literal[
    'view_workspace',
    'show_details',
    'view_file_content',
    'edit_file_content',
    'str_replace_edit',
    'undo',
    'search',
    'operate_filesystem'
      # 'create'
      # 'remove'
      # 'rename'
]

class EditTools(ToolCollection):
    SCHEMA_FILE: str = 'schema_edit_collections'
    SKILL_FILE: str = 'skill_edit_collections'

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
            'str_replace_edit': 'str_replace_edit',
            'undo': 'undo',
            'search': 'search',
            'operate_filesystem': 'operate_filesystem'
        },
        add_skill: bool = True
    ) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex, { value: f"_tool_{key}" for key, value in rename_mapping.items() })
        
        self.client = client
        self.permission_required = permission_required
        self.names = rename_mapping
        self.add_skill = add_skill

        self.base_dir: Path = Path(base_dir)
        self.initialized: bool = False

        self.tool_definitions = load_tool_definitions(self.SCHEMA_FILE)
        self.view_workspace_schema = load_schema(self.SCHEMA_FILE, 'view_workspace', self.names.get('view_workspace', 'view_workspace'))
        self.show_details_schema = load_schema(self.SCHEMA_FILE, 'show_details', self.names.get('show_details', 'show_details'))
        self.view_file_content_schema = load_schema(self.SCHEMA_FILE, 'view_file_content', self.names.get('view_file_content', 'view_file_content'))
        self.edit_file_content_schema = load_schema(self.SCHEMA_FILE, 'edit_file_content', self.names.get('edit_file_content', 'edit_file_content'))
        self.str_replace_edit_schema = load_schema(self.SCHEMA_FILE, 'str_replace_edit', self.names.get('str_replace_edit', 'str_replace_edit'))
        self.undo_schema = load_schema(self.SCHEMA_FILE, 'undo', self.names.get('undo', 'undo'))
        self.search_schema = load_schema(self.SCHEMA_FILE, 'search', self.names.get('search', 'search'))
        self.operate_filesystem_schema = load_schema(self.SCHEMA_FILE, 'operate_filesystem', self.names.get('operate_filesystem', 'operate_filesystem'))
          # self.create_schema = load_schema(self.SCHEMA_FILE, 'create', self.names.get('create', 'create'))
          # self.remove_schema = load_schema(self.SCHEMA_FILE, 'remove', self.names.get('remove', 'remove'))
          # self.rename_schema = load_schema(self.SCHEMA_FILE, 'rename', self.names.get('rename', 'rename'))

        self.all_schemas: Dict[EditOperation, ToolSchema] = {
            'view_workspace': self.view_workspace_schema,
            'show_details': self.show_details_schema,
            'view_file_content': self.view_file_content_schema,
            'edit_file_content': self.edit_file_content_schema,
            'str_replace_edit': self.str_replace_edit_schema,
            'undo': self.undo_schema,
            'search': self.search_schema,
            'operate_filesystem': self.operate_filesystem_schema
        }

        self.schemas: Dict[EditOperation, ToolSchema] = { key: value for key, value in self.all_schemas.items() if key in self.names }
        self.skill: str = load_tool_skill(self.SKILL_FILE, {str(k): f"`{v}`" for k, v in self.names.items()})
        self.skill_added: bool = False

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

    def clone(self) -> "EditTools":
        raise RuntimeError(f"{self.__class__.__name__} is not safeply copyable")

    async def bind_io(self, input_interface: UserInputInterface, **kwargs) -> None:
        self.input_interface = input_interface

    def process_prompt(self, user_prompt: PromptTemplate, **kwargs) -> Optional[AgentLoopStateEdit]:
        if not self.skill_added and self.add_skill:
            self.skill_added = True
            new_user_prompt = user_prompt + self.skill
            return AgentLoopStateEdit(user_prompt = new_user_prompt)

    async def start_loop_async(self, *args, **kwargs) -> None:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        query: Dict = { 'type': 'refresh' }

        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
        except Exception:
            raise

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
        
        if line_start and line_start < 0:
            return f"[ERROR]: Parameter 'line_start' should be non-negative!"
        if line_end and line_end < 0:
            return f"[ERROR]: Parameter 'line_end' should be non-negative!"
        if line_start and line_end and line_end < line_start:
            return f"[ERROR]: Parameter 'line_end' should be greater or equal to 'line_start'!"
        
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
        
        if line_start and line_start < 0:
            return f"[ERROR]: Parameter 'line_start' should be non-negative!"
        if line_end and line_end < 0:
            return f"[ERROR]: Parameter 'line_end' should be non-negative!"
        if line_start and line_end and line_end < line_start:
            return f"[ERROR]: Parameter 'line_end' should be greater or equal to 'line_start'!"
        
        if edit_type not in ['replace', 'insert']:
            return f"[ERROR]: Parameter 'edit_type' should be one of 'replace' or 'insert'."
        
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

    async def _tool_str_replace_edit(
        self,
        target_path: str,
        original_content: str,
        new_content: str,
        scope: str,
        **kwargs
    ) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        if scope not in ['once_only', 'all']:
            return f"[ERROR]: Parameter 'scope' should be one of 'once_only' or 'all'."
        
        query: Dict = {
            'type': 'str_replace_edit',
            'target_path': target_path,
            'original_content': original_content,
            'new_content': new_content,
            'replace_all': True if scope == 'all' else False
        }

        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise

    async def _tool_undo(
        self,
        target_path: str,
        **kwargs
    ) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        query: Dict = {
            'type': 'undo',
            'target_path': target_path,
        }

        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise
    
    async def _tool_search(self, key_words: str, glob: str, mode: str, **kwargs) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        if mode not in ['definition', 'identifier', 'pattern']:
            return f"[ERROR]: Parameter 'mode' should be one of 'definition', 'identifier' or 'pattern'."
        
        notice: str = ""
        if mode in ['definition', 'identifier']:
            keyword_list = [word.strip() for word in key_words.split(',')]
            valid_id_regex = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
            invalid_keywords = []
            
            for keyword in keyword_list:
                if not keyword or not valid_id_regex.match(keyword):
                    invalid_keywords.append(keyword)
            
            if invalid_keywords:
                invalid_str = ", ".join(invalid_keywords)
                notice = (
                    f"[NOTICE] The input keywords '{invalid_str}' do not conform to standard identifier rules "
                    f"(letters, digits, underscores only, cannot start with a digit). "
                    f"The search may miss results in {mode} mode.\n"
                )

        query: Dict = {
            'type': 'search_entity',
            'key_words': [word.strip() for word in key_words.split(',')] if mode != 'pattern' else key_words.strip(),
            'glob': glob,
            'mode': mode
        }
        
        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return notice + response.strip()
        except Exception:
            raise

    async def _tool_operate_filesystem(self, operation: str, target_path: str, **kwargs) -> str:
        if not self.initialized:
            raise UnbuiltError(self.__class__.__name__)
        
        if operation not in ['create', 'remove', 'rename']:
            return f"[ERROR]: Parameter 'operation' should be one of 'create', 'remove' or 'rename'."
        
        if operation == 'create':
            if self.input_interface and self.permission_required:
                user_response = await self.input_interface.notify_user(UserNotify('permission', f"Do you allow agent to create file: {target_path}?"))
                if not user_response.permitted:
                    return f"[ERROR]: Permission error! {user_response.reason}"
                
            if 'content' in kwargs:
                content: str = kwargs.get('content') # type: ignore
            else:
                return f"[ERROR]: Missing argument 'content' for 'create' operation to initialize file content."
        
            query: Dict = {
                'type': 'touch',
                'target_path': target_path,
                'content': content
            }
        elif operation == 'remove':
            if self.input_interface and self.permission_required:
                user_response = await self.input_interface.notify_user(UserNotify('permission', f"Do you allow agent to remove files or directories: {target_path}?"))
                if not user_response.permitted:
                    return f"[ERROR]: Permission error! {user_response.reason}"
                
            query: Dict = {
                'type': 'remove',
                'target_path': target_path
            }
        elif operation == 'rename':
            if 'source_path' in kwargs:
                source_path = kwargs.get('source_path')
            else:
                return f"[ERROR]: Missing argument 'source_path' for 'rename' operation."

            if self.input_interface and self.permission_required:
                user_response = await self.input_interface.notify_user(UserNotify('permission', f"Do you allow agent to move {source_path} to {target_path}?"))
                if not user_response.permitted:
                    return f"[ERROR]: Permission error! {user_response.reason}"
            
            query: Dict = {
                'type': 'rename',
                'src_path': source_path,
                'dst_path': target_path
            }
        else:
            return f"[ERROR]: Unsupported operation specified: '{operation}'."

        try:
            response: str = await self.client.exchange(json.dumps(query))
            if response is None:
                raise RequestError(content = f'unable to access {self.client.url}')
            return response.strip()
        except Exception:
            raise

if __name__ == '__main__':
    pass
