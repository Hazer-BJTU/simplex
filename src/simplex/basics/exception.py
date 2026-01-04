from typing import Optional

class CustomException(Exception):
    pass

class EntityInitializationError(CustomException):
    def __init__(self, name: str, original: Exception) -> None:
        self.name = name
        self.original = original
        super().__init__(f"failed to initialize {name} due to exception: {self.original}")

class RequestError(CustomException):
    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(f"request error due to exception: {self.original}")

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

if __name__ == '__main__':
    pass
