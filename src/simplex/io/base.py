import os

from abc import ABC, abstractmethod
from typing import Optional, Any

import simplex.basics

from simplex.basics import UserMessage, UserNotify, UserResponse
from simplex.context.base import ContextPlugin


class UserInputInterface(ABC):
    """
    Abstract base class defining the interface for user input in the agent framework.
    
    This interface is responsible for capturing user input and handling interactive
    notifications (e.g., permission requests). It is used by the UserLoop to obtain
    user messages and by tools/plugins to request user permissions during execution.
    
    Concrete implementations define how input is acquired (e.g., terminal, GUI, API).
    
    The interface also optionally provides a ContextPlugin that can be added to the
    AgentLoop to inject input‑related context or modify the loop's behavior.
    
    See also:
        UserLoop: The outer loop that drives the conversation using this interface.
        AgentLoop: The inner loop that uses notify_user for permission requests.
        RichTerminalInterface: A concrete implementation for terminal interaction.
    """
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    async def next_message(self) -> UserMessage:
        """
        Asynchronously retrieve the next user message.
        
        This method is called by the UserLoop at the beginning of each iteration
        to obtain the user's input. The returned UserMessage contains:
        - system_prompt: PromptTemplate for the system role.
        - user_prompt: PromptTemplate for the user role.
        - quit: Boolean flag indicating that the user wishes to exit the loop.
        
        When quit is True, the loop terminates and no further processing occurs.
        The implementation should handle input acquisition, any preprocessing,
        and optionally integrate skill retrieval or other input enhancements.
        
        Returns:
            UserMessage: A structured message containing prompts and quit flag.
        
        Raises:
            Implementation‑specific exceptions may be raised (e.g., IO errors,
            unexpected input format). These will be caught by the loop's exception
            handler.
        """
        pass

    @abstractmethod
    async def notify_user(self, notify: UserNotify) -> UserResponse:
        """
        Present a notification to the user and wait for a response.
        
        This method is primarily used for permission requests (notify_type='permission')
        but can also handle generic notifications. The implementation should display
        the notification content (title, text, objects) in a user‑appropriate manner
        and capture the user's decision.
        
        For permission requests, the user's response (permitted=True/False) and an
        optional reason must be returned. For other notification types, the default
        response should indicate permission granted (or a suitable placeholder).
        
        Args:
            notify: UserNotify instance describing the notification.
                - notify_type: 'permission', 'notify', or 'unknown'.
                - content: Main text of the notification.
                - title: Optional title for the notification.
                - objects: Optional list of dictionaries to display as a table.
        
        Returns:
            UserResponse: Contains permitted flag and optional reason string.
        
        Raises:
            Implementation‑specific exceptions may be raised (e.g., communication
            errors, timeouts). These will be caught by the loop's exception handler.
        """
        pass

    @abstractmethod
    def get_input_plugin(self) -> Optional[ContextPlugin]:
        """
        Retrieve an optional ContextPlugin associated with this input interface.
        
        Some input interfaces may provide a plugin that adds input‑related context
        to the AgentLoop (e.g., a plugin that modifies prompts based on user
        identity or input history). If no such plugin is needed, return None.
        
        The plugin will be automatically added to the AgentLoop when bind_io is
        called, before the loop's build phase.
        
        Returns:
            Optional[ContextPlugin]: A context plugin instance, or None.
        """
        pass
    

class UserOutputInterface(ABC):
    """
    Abstract base class defining the interface for user output in the agent framework.
    
    This interface is responsible for presenting information to the user, such as
    agent responses, tool execution results, and system notifications. It is used
    by the AgentLoop (via push_message) and by context plugins/tools that need to
    send notifications to the user.
    
    Concrete implementations define how output is rendered (e.g., terminal, GUI,
    logging service).
    
    The interface also optionally provides a ContextPlugin that can be added to the
    AgentLoop to inject output‑related context or modify the loop's behavior.
    
    See also:
        AgentLoop: The inner loop that uses push_message to send notifications.
        RichTerminalInterface: A concrete implementation for terminal output.
        UserLoop: The outer loop that may also use output for status messages.
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    async def push_message(self, notify: UserNotify) -> Any:
        """
        Asynchronously send a notification to the user.
        
        This method is called by the AgentLoop and its plugins/tools to present
        information to the user. The notification can be of any type (e.g., 'notify',
        'permission', 'unknown'). Unlike notify_user, push_message does not wait for
        a response; it is a one‑way communication.
        
        The implementation should render the notification content (title, text,
        objects) appropriately. For example, a terminal interface might display a
        styled panel, while a GUI might show a dialog.
        
        Args:
            notify: UserNotify instance describing the notification.
                - notify_type: 'permission', 'notify', or 'unknown'.
                - content: Main text of the notification.
                - title: Optional title for the notification.
                - objects: Optional list of dictionaries to display as a table.
        
        Returns:
            Any: Implementation‑specific return value (may be ignored).
        
        Raises:
            Implementation‑specific exceptions may be raised (e.g., rendering errors,
            communication failures). These will be caught by the loop's exception
            handler.
        """
        pass

    @abstractmethod
    def get_output_plugin(self) -> Optional[ContextPlugin]:
        """
        Retrieve an optional ContextPlugin associated with this output interface.
        
        Some output interfaces may provide a plugin that adds output‑related context
        to the AgentLoop (e.g., a plugin that logs all outgoing notifications or
        adds formatting rules). If no such plugin is needed, return None.
        
        The plugin will be automatically added to the AgentLoop when bind_io is
        called, before the loop's build phase.
        
        Returns:
            Optional[ContextPlugin]: A context plugin instance, or None.
        """
        pass

if __name__ == '__main__':
    pass
