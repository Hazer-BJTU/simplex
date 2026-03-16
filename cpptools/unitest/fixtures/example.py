from typing import Any


decimal: int = 10
point: float = 1.0

def function(*args, **kwargs) -> bool:
    return True

my_variable: bool = function([], {}, {'a': 1, 'b': 2, 'c': 3})

class MyClass(Exception):
    my_variable: bool = True

    class MyInnerClass(Exception):
        my_variable: bool = False

        def method(self, *args, **kwargs) -> Any:
            return {}

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

if __name__ == '__main__':
    func2 = function
    x = func2(decimal, point)
    
    y = MyClass.MyInnerClass().method()
