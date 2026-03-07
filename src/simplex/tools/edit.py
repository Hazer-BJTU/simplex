import os
import json
import uuid

from enum import Enum, auto
from typing import Optional, List, Dict, Callable, TYPE_CHECKING

import simplex.basics
import simplex.tools.base

from simplex.basics import (
    ModelInput,
    WebsocketClient,
    UnbuiltError,
    RequestError
)
from simplex.tools.base import ToolCollection

if TYPE_CHECKING:
    import simplex.loop

    from simplex.loop import AgentLoop


class EditTools(ToolCollection):
    class Operation(Enum):
        view_workspace = auto()
        show_details = auto()
        view_file_content = auto()
        edit_file_content = auto()
        search = auto()
        create_file = auto()
        remove_file = auto()

    VIEW_WORKSPACE = Operation.view_workspace.name
    SHOW_DETAILS = Operation.show_details.name
    VIEW_FILE_CONTENT = Operation.view_file_content.name
    EDIT_FILE_CONTENT = Operation.edit_file_content.name
    SEARCH = Operation.search.name
    CREATE_FILE = Operation.create_file.name
    REMOVE_FILE = Operation.remove_file.name

    def __init__(
        self,
        client: WebsocketClient,
        instance_id: str = uuid.uuid4().hex, 
        rename_mapping: Dict[str, str] = {
            VIEW_WORKSPACE: 'view_workspace',
            SHOW_DETAILS: 'show_details',
            VIEW_FILE_CONTENT: 'view_file_content',
            EDIT_FILE_CONTENT: 'edit_file_content',
            SEARCH: 'search',
            CREATE_FILE: 'create_file',
            REMOVE_FILE: 'remove_file'
        }
    ) -> None:
        super().__init__(instance_id, { value: f"_tool_{key}" for key, value in rename_mapping.items() })
        
        self.client = client
        self.names = rename_mapping
        self.initialized: bool = False

        self.view_workspace_schema = {
            "type": "function",
            "function": {
                "name": self.names[self.VIEW_WORKSPACE],
                "description": "Used to retrieve the current workspace file structure. " \
                               "In some editing operations, updated workspace information may be provided directly, " \
                               "eliminating the need for repeated retrieval.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        self.show_details_schema = {
            "type": "function",
            "function": {
                "name": self.names[self.SHOW_DETAILS],
                "description": "Used to display detailed information about a directory or file in the workspace. " \
                               "For a directory, list all its subdirectories and files; for a file, attempt to analyze its source code structure or provide a content preview. " \
                               "You can navigate directly to the corresponding target without concerning current the workspace view. " \
                               "It is a good practice to use this method to show a structural preview before browsing the file content directly.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Please use the relative path in the workspace!"
                        }
                    },
                    "required": ["target_path"]
                }
            }
        }

        self.view_file_content_schema = {
            "type": "function",
            "function": {
                "name": self.names[self.VIEW_FILE_CONTENT],
                "description": "Used to browse the specific content of a file, with each line of the file displayed with a line number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Please use the relative path in the workspace!"
                        },
                        "line_start": {
                            "type": "int",
                            "description": "Indicates the starting line number, where line numbering begins at 1."
                        },
                        "line_end": {
                            "type": "int",
                            "description": "Indicates the ending line number (inclusive)."
                        }
                    },
                    "required": ["target_path"]
                }
            }
        }

        self.edit_file_content_schema = {
            "type": "function",
            "function": {
                "name": self.names[self.EDIT_FILE_CONTENT],
                "description": "Used for editing text content. There are two modes: \'replace\' and \'insert\'. " \
                               "In \'insert\' mode, the \'content\' is inserted \'after\' the specified \'line_start\' line number. " \
                               "In \'replace\' mode, the \'content\' between line numbers [\'line_start\', \'line_end\'] inclusive is replaced with \'content\'. " \
                               "You can judge whether the edit is correct based on the feedback. " \
                               "Pay attention to indentation alignment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Please use the relative path in the workspace!"
                        },
                        "edit_type": {
                            "type": "string",
                            "description": "\'replace\' or \'insert\'"
                        },
                        "content": {
                            "type": "string",
                            "description": "Pay special attention to indentation when writing content, " \
                                           "especially for languages like Python that enforce indentation alignment."
                        },
                        "line_start": {
                            "type": "int",
                            "description": "Indicates the starting line number, where line numbering begins at 1."
                        },
                        "line_end": {
                            "type": "int",
                            "description": "Indicates the ending line number (inclusive). This parameter is ignored in \'insert\' mode."
                        }
                    },
                    "required": ["target_path", "edit_type", "content"]
                }
            }
        }

        self.search_schema = {
            "type": "function",
            "function": {
                "name": self.names[self.SEARCH],
                "description": "Used for keyword retrieval. " \
                               "For a given set of keywords, \'semantic_search\' attempts to search within code elements (such as class names, method names). " \
                               "If no matches are found, try to fall back to \'pattern_match\' retrieval. " \
                               "During the retrieval process, try to avoid using overly broad keywords (e.g., '__init__'). " \
                               "Two retrieval scopes are supported: \'workspace\' and \'global\'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key_words": {
                            "type": "string",
                            "description": "Given a set of keywords separated by commas, e.g., \'vector, matrix, tensor\'."
                        },
                        "scope": {
                            "type": "string",
                            "description": "Choose between \'global\' or \'workspace\'. " \
                                           "The \'workspace\' scope only looks for files explicitly listed in a tree structure within the most recent workspace from the history. " \
                                           "The \'global\' scope is independent of the workspace view. " \
                                           "Use \'global\' unless you are certain about the search scope. "
                        },
                        "mode": {
                            "type": "string",
                            "description": "Choose between \'semantic_search\' or \'pattern_match\'. " \
                                           "The \'semantic_search\' attempts to search within code elements. " \
                                           "The \'pattern_match\' only performs pattern matching. "
                        }
                    },
                    "required": ["key_words", "scope", "mode"]
                }
            }
        }
        
        self.create_file_schema = {
            "type": "function",
            "function": {
                "name": self.names[self.CREATE_FILE],
                "description": "Used for \'touch\' a new file with initial content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Please use the relative path in the workspace!"
                        },
                        "content": {
                            "type": "string",
                            "description": "Initial content will be written into the file."
                        }
                    },
                    "required": ["target_path", "content"]
                }
            }
        }

        self.remove_file_schema = {
            "type": "function",
            "function": {
                "name": self.names[self.REMOVE_FILE],
                "description": "Used for \'rm\' a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Please use the relative path in the workspace!"
                        }
                    },
                    "required": ["target_path"]
                }
            }
        }

        self.schemas: Dict[str, Dict] = {
            self.VIEW_FILE_CONTENT: self.view_file_content_schema,
            self.SHOW_DETAILS: self.show_details_schema,
            self.VIEW_FILE_CONTENT: self.view_file_content_schema,
            self.EDIT_FILE_CONTENT: self.edit_file_content_schema,
            self.SEARCH: self.search_schema,
            self.CREATE_FILE: self.create_file_schema,
            self.REMOVE_FILE: self.remove_file_schema
        }

    async def build(self) -> None:
        try:
            await self.client.build()
            self.initialized = True
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

    def get_names(self) -> List[str]:
        return list(self.names.values())

    def get_tools(self) -> List[Dict]:
        return list(self.schemas.values())
    
    def tools_descriptions(self) -> List[Dict]:
        return list(self.schemas.values())
    
    def on_init_output(self, model_input: ModelInput, agent: "AgentLoop") -> None:
        pass

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
    
    async def _tool_create_file(self, target_path: str, content: str, **kwargs) -> str:
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
    
    async def _tool_remove_file(self, target_path: str, **kwargs) -> str:
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
