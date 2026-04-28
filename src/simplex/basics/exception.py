import os
import sys
import copy
import uuid

from datetime import datetime
from abc import ABC, abstractmethod
from typing import Optional, List, Any


class CustomException(Exception):
    pass

class EntityInitializationError(CustomException):
    def __init__(self, name: str, original: Exception) -> None:
        self.name = name
        self.original = original
        super().__init__(f"failed to initialize {name} due to exception: {self.original}")

class RequestError(CustomException):
    def __init__(self, original: Optional[Exception] = None, content: str = '') -> None:
        self.original = original
        if original is not None:
            super().__init__(f"request error due to exception: {self.original}")
        else:
            super().__init__(F"{content}")

class ParameterError(CustomException):
    def __init__(
        self,
        function_name: str,
        parameter: str,
        content: str,
        type_hint: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> None:
        description: str = ''
        if class_name is not None:
            description += f"{class_name}."
        description += f"{function_name}({parameter}"
        if type_hint is not None:
            description += f": {type_hint}"
        description += f"): {content}"
        super().__init__(description)

class ImplementationError(CustomException):
    def __init__(
        self,
        function_name: str,
        content: str,
        class_name: Optional[str] = None
    ) -> None:
        description: str = ''
        if class_name is not None:
            description += f"{class_name}."
        description += f"{function_name}: {content}"
        super().__init__(description)

class EnvironmentError(CustomException):
    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(f"failed to initialize or release environment due to: {self.original}")

class UnbuiltError(CustomException):
    def __init__(self, class_name: str) -> None:
        super().__init__(f"{class_name}: method 'build' is never called") 

class ConflictError(CustomException):
    def __init__(self, content: str = '') -> None:
        super().__init__(content)

class MaxRetriesExceeded(CustomException):
    def __init__(self, content: str = '') -> None:
        super().__init__(content)

class Notice(CustomException):
    def __init__(self, content: str = '') -> None:
        super().__init__(content)

class ExceptionHandler(ABC):
    """
    Handles exceptions generated during the agent loop execution.
    
    This class is designed to process exceptions that occur in the context of
    sequential/asynchronous method calls of components such as ContextPlugin and
    ToolCollection. It can extract and handle exceptions from a result list
    (similar to the return value of asyncio.gather), as well as process individual exceptions.
    """

    def __init__(self, instance_id: str) -> None:
        super().__init__()
        self.__instance_id = instance_id

    @property
    def key(self) -> str:
        return self.__instance_id

    @abstractmethod
    def handle_exception(self, exception: Exception) -> None:
        """
        Process a single exception instance (to be implemented by subclasses).

        This abstract method must be overridden by any subclass of ExceptionHandler.
        It defines the specific logic for handling an individual exception.

        Args:
            exception: The single Exception instance to be processed
        """
        pass

    def __call__(self, *args: Exception | List) -> None:
        for arg in args:
            if isinstance(arg, Exception):
                self.handle_exception(arg)
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, Exception):
                        self.handle_exception(item)

    def clone(self) -> "ExceptionHandler":
        return copy.deepcopy(self)

class LogExceptionHandler(ExceptionHandler):
    def __init__(self, instance_id: Optional[str] = None, content: str = '', file: Any = sys.stderr) -> None:
        super().__init__(instance_id if instance_id is not None else uuid.uuid4().hex)
        self.content: str = content
        self.file = file

    def handle_exception(self, exception: Exception) -> None:
        """
        Process a single exception instance with timestamped logging.

        This concrete implementation formats the exception with current timestamp,
        prints the error message to standard error stream (stderr), and appends
        the formatted message to the instance's `content` attribute for persistent
        storage/audit purposes.

        Args:
            exception: The single Exception instance to be processed and logged
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{current_time}] {type(exception).__name__}: {exception}\n"
        if self.file:
            print(message, file = self.file)
        self.content += message

    def clone(self) -> "LogExceptionHandler":
        return LogExceptionHandler(
            instance_id = uuid.uuid4().hex, 
            content = self.content, 
            file = self.file
        )

    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        return f"LogExceptionHandler(key={repr(self.key)}, content={repr(self.content)})"

if __name__ == '__main__':
    pass
