import os
import json

from typing import Optional, List, Dict, Any


class PromptTemplate:
    def __init__(self) -> None:
        self.content: str = ''

    def add_main_title(self, title: str) -> "PromptTemplate":
        self.content += f"# {title.strip()}\n\n"
        return self
    
    def add_sub_title(self, title: str) -> "PromptTemplate":
        self.content += f"## {title.strip()}\n\n"
        return self

    def add_simple(self, text: str | List[str], title: Optional[str] = None) -> "PromptTemplate":
        if isinstance(text, str):
            flattened: str = text.strip()
        elif isinstance(text, List):
            flattened: str = '\n\n'.join([component.strip() for component in text])
        
        if title is not None:
            self.content += f"## {title}\n\n{flattened}\n\n"
        else:
            self.content += f"{flattened}\n\n"
        return self
    
    def add_block(self, text: str | List[str], title: Optional[str] = None, block: str = '', as_whole: bool = False) -> "PromptTemplate":
        if isinstance(text, str):
            flattened: str = f"`````{block}\n{text.strip()}\n`````"
        elif isinstance(text, List):
            if as_whole:
                flattened: str = '\n\n'.join([component.strip() for component in text])
                flattened = f"`````{block}\n{flattened}\n`````"
            else:
                flattened: str = '\n\n'.join([f"`````{block}\n{component.strip()}\n`````" for component in text])

        if title is not None:
            self.content += f"## {title}\n\n{flattened}\n\n"
        else:
            self.content += f"{flattened}\n\n"
        return self
    
    def __str__(self) -> str:
        return self.content.strip()
    
    def __repr__(self) -> str:
        return f"PromptTemplate({repr(self.content.strip())})"

if __name__ == '__main__':
    pass
