import os

from typing import Optional

import simplex.basics.exception
import simplex.basics.dataclass
import simplex.tools.base

from simplex.tools.base import ToolCollection


class PythonInterpreter(ToolCollection):
    def __init__(
        self, 
        rename: str = 'python_interpreter',
        use_container: bool = False,
        container_id: Optional[str] = None,
        default_image_name: Optional[str] = None,
        auto_pull: bool = True
    ) -> None:
        self.name = rename
        self.pyinterpreter_schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "execute given python script and return program output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "python script, e.g., 'import math; print(math.sqrt(math.sin(math.pi)))'",
                        }
                    },
                    "required": ["script"],
                },
            },
        }

    


if __name__ == '__main__':
    pass
