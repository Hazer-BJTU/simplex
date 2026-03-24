import os
import re
import json
import yaml
import glob
import pathlib
import numpy as np

from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import Optional, List, Dict, Any


MODULE_PATH: Path = Path(__file__).resolve().parent
SKILLS_PATH: Path = MODULE_PATH / 'presets' / 'skills'
SYSTEM_PATH: Path = MODULE_PATH / 'presets' / 'system'

class PromptTemplate:
    def __init__(self, content: str = '') -> None:
        self.content = content

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
    
    def __add__(self, other):
        new_content: str = f"{self.content.strip()}\n\n{str(other)}"
        return PromptTemplate(new_content)

    def __radd__(self, other):
        new_content: str = f"{str(other)}\n\n{self.content.strip()}"
        return PromptTemplate(new_content)

    def __iadd__(self, other):
        self.content = f"{self.content.strip()}\n\n{str(other)}"
        return self
    
class SkillRetriever:
    def __init__(self, top_k: int = 5, path: Path = SKILLS_PATH) -> None:
        self.top_k = top_k
        self.path = path

        self.skills: List[Dict] = []
        self.corpus: List[List[str]] = []
        self._load_all_skills()

        self.bm25 = BM25Okapi(self.corpus)
        self.sorted_indices: Optional[np.ndarray] = None

    def _load_all_skills(self) -> None:
        if not os.path.exists(self.path):
            return
        
        for file_path in self.path.rglob(r"*.yml"):
            try:
                with open(file_path, 'r', encoding = 'utf8') as file:
                    skill_data = yaml.safe_load(file)

                title = skill_data.get('title', '')
                description = skill_data.get('description', '...')
                tags = skill_data.get('tags', [])
                content = skill_data.get('content', '')

                if not all([title, description, tags,  content]):
                    continue

                search_text = f"{title}: {' '.join(tags)} {description} {content}".lower()
                words = re.findall(r"[a-zA-Z']+", search_text)

                self.skills.append({'title': title, 'description': description, 'tags': ' '.join(tags), 'content': content})
                self.corpus.append(words)
            except Exception:
                continue

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        if not top_k:
            top_k = self.top_k

        tokenized_query: List[str] = re.findall(r"[a-zA-Z']+", query)
        scores = self.bm25.get_scores(tokenized_query)

        scores = np.array(scores)
        self.sorted_indices = np.argsort(scores)[::-1]
        selected_indices = self.sorted_indices[:top_k]

        return [self.skills[idx] for idx in selected_indices]
    
    def get_more(self, top_k: int) -> List[Dict]:
        if self.sorted_indices:
            selected_indices = self.sorted_indices[:top_k]
            return [self.skills[idx] for idx in selected_indices]
        else:
            return []
    
    def get_system_prompt(self, path: Optional[Path] = None) -> PromptTemplate:
        if not path:
            path = SYSTEM_PATH / 'general_develop.md'

        with open(path, 'r', encoding = 'utf8') as file:
            content = file.read()

        return PromptTemplate(content)

if __name__ == '__main__':
    pass
