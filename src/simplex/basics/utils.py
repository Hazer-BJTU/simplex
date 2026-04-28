import os
import asyncio

from functools import wraps
from typing import Any, Callable, Type, Tuple

import simplex.basics.exception

from simplex.basics.exception import MaxRetriesExceeded


def async_retry_timeout(
    max_retry: int,
    timeout: float,
    retry_delay: float = 0.5,
    retry_exceptions: Tuple[Type[Exception], ...] = (asyncio.TimeoutError,),
    on_retry: Callable[[Exception, int], None] = lambda e, attempt: print(f"retry attempt [{attempt + 1}] {type(e).__name__}: {e}")
) -> Callable:
    """
    Async retry & timeout control decorator.
    Adds automatic timeout limitation and exception-retry logic for async functions.
    Only catches specified exceptions to trigger retries.

    Args:
        max_retry: Maximum retry count (initial execution not counted; total runs = max_retry + 1)
        timeout: Timeout threshold in seconds for single async function execution
        retry_delay: Delay in seconds before each retry, default 0.5s
        retry_exceptions: Tuple of exception types that will trigger retry
        on_retry: Callback function executed on each retry; prints log by default.
                  Params: exception instance, current attempt index, max retry limit

    Returns:
        Wrapped async function with original signature preserved
    """
    def decorator(func: Callable) -> Callable:
        """Inner decorator that accepts the target async function"""
        @wraps(func)
        async def inner_function(*args: Any, **kwargs: Any) -> Any:
            """
            Core wrapper logic for retry and timeout management.
            Args:
                *args: Positional arguments passed to the original function
                **kwargs: Keyword arguments passed to the original function

            Returns:
                Return value of the original executed async function

            Raises:
                MaxRetriesExceeded: Raised when all retry attempts fail
                Exception: Non-retryable exceptions will be re-raised directly
            """
            # Total execution rounds: initial run + configured retries
            for attempt in range(max_retry + 1):
                try:
                    # Enforce single execution timeout with asyncio.wait_for
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout = timeout
                    )

                # Catch designated exceptions for retry processing
                except retry_exceptions as e:
                    # Terminate and raise custom exception if retries are exhausted
                    if attempt >= max_retry:
                        raise MaxRetriesExceeded(f"maximum retry limit {max_retry} reached, task failed permanently") from e
                    
                    # Trigger custom retry callback
                    on_retry(e, attempt)
                    # Pause before next retry attempt
                    await asyncio.sleep(retry_delay)

                # Re-raise unexpected exceptions without retry
                except Exception:
                    raise
        
        inner_function.retry_already_handled = True # type: ignore
        return inner_function
    return decorator

if __name__ == '__main__':
    pass
