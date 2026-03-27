import os
import uuid

from typing import Dict, List, Any
from openai import AsyncOpenAI
from abc import ABC, abstractmethod

import simplex.basics
import simplex.tools

from simplex.basics import (
    ToolSchema,
    ModelInput, 
    ModelResponse, 
    DocumentEntry, 
    ToolReturn, 
    EntityInitializationError
)


class BaseModel(ABC):
    """Abstract base class for all AI models in the Simplex framework.

    This class defines the common interface and initialization logic for
    AI models, including configuration management and OpenAI client setup.

    Attributes:
        key (str): Unique instance identifier (read-only property).
        _base_url (str): Base URL for the model API (protected property).
        _api_key (str): API key for authentication (protected property).
        _client_configs (Dict): Additional client configuration (protected).
        _default_generate_configs (Dict): Default generation parameters (protected).

    Args:
        base_url (str): Base URL for the model API.
        api_key (str): API key for authentication.
        client_configs (Dict): Additional configuration for the AsyncOpenAI client.
        default_generate_configs (Dict): Default parameters for generation calls.
        instance_id (str): Unique identifier for this model instance.
        disable_openai_backend (bool, optional): If True, skip OpenAI client
            initialization. Defaults to False.

    Raises:
        EntityInitializationError: If client initialization fails.
    """
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        client_configs: Dict, 
        default_generate_configs: Dict,
        instance_id: str,
        disable_openai_backend: bool = False
    ) -> None:
        """Initialize the BaseModel with configuration.

        See class docstring for parameter details.
        """
        super().__init__()
        self.__base_url = base_url
        self.__api_key = api_key
        self.__client_configs = client_configs
        self.__default_generate_configs = default_generate_configs
        self.__instance_id = instance_id
        self.__disable_openai_backend = disable_openai_backend

        try:
            if not self.__disable_openai_backend:
                self.client = AsyncOpenAI(
                    base_url = self.__base_url, 
                    api_key = self.__api_key,
                    **self.__client_configs
                )
            else:
                self.client = None
        except Exception as e:
            raise EntityInitializationError(self.__class__.__name__, e)
        
    @property
    def key(self) -> str:
        """Unique instance identifier for this model.

        Returns:
            str: The instance ID provided during initialization.
        """
        return self.__instance_id
    
    @property
    def _base_url(self) -> str:
        """Base URL for the model API (protected).

        Returns:
            str: The base URL provided during initialization.
        """
        return self.__base_url
    
    @property
    def _api_key(self) -> str:
        """API key for authentication (protected).

        Returns:
            str: The API key provided during initialization.
        """
        return self.__api_key
    
    @property
    def _client_configs(self) -> Dict:
        """Additional client configuration dictionary (protected).

        Returns:
            Dict: The client configuration dictionary provided during initialization.
        """
        return self.__client_configs
    
    @property
    def _default_generate_configs(self) -> Dict:
        """Default generation configuration dictionary (protected).

        Returns:
            Dict: The default generation parameters provided during initialization.
        """
        return self.__default_generate_configs
    
    @abstractmethod
    def clone(self) -> "BaseModel":
        """Create a copy of this model instance.

        Returns:
            BaseModel: A new instance with the same configuration.
        """
        pass
        
    @abstractmethod
    async def build(self) -> None:
        """Prepare the model for use (e.g., load resources, establish connections).

        This method should be called before using the model for generation.
        """
        pass

    @abstractmethod
    async def release(self) -> None:
        """Release resources and clean up after model use.

        This method should be called when the model is no longer needed.
        """
        pass

    @abstractmethod
    async def reset(self) -> None:
        """Reset the model's internal state (e.g., clear conversation history).

        This method should be called to return the model to a clean state.
        """
        pass

    @abstractmethod
    async def generate(self, model_input: ModelInput) -> ModelResponse:
        """Generate a response from the model.

        Args:
            model_input (ModelInput): Input data for the generation.

        Returns:
            ModelResponse: The model's response.
        """
        pass

class EmbeddingModel(BaseModel):
    """Abstract base class for embedding models.

    Embedding models convert documents into vector representations.
    This class provides default implementations for most BaseModel methods,
    leaving batch_embedding as the only abstract method that must be implemented.

    Note:
        The default implementations of clone, build, release, reset, and generate
        are minimal and should be overridden by concrete subclasses.
    """
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        client_configs: Dict, 
        default_generate_configs: Dict,
        instance_id: str,
        disable_openai_backend: bool = False
    ) -> None:
        """Initialize an embedding model.

        Args:
            base_url (str): Base URL for the embedding API.
            api_key (str): API key for authentication.
            client_configs (Dict): Additional client configuration.
            default_generate_configs (Dict): Default generation parameters.
            instance_id (str): Unique identifier for this instance.
            disable_openai_backend (bool, optional): Whether to disable OpenAI backend. Defaults to False.
        """
        super().__init__(
            base_url, 
            api_key, 
            client_configs, 
            default_generate_configs, 
            instance_id,
            disable_openai_backend
        )

    def clone(self) -> "EmbeddingModel":
        """Create a copy of this embedding model instance (placeholder).

        Note:
            This is a placeholder implementation that returns None.
            Concrete subclasses should override this method.

        Returns:
            EmbeddingModel: A new instance with the same configuration.
        """
        return None # type: ignore

    async def build(self) -> None:
        """Prepare the embedding model for use (placeholder).

        Note:
            This is a placeholder implementation that does nothing.
            Concrete subclasses should override this method.
        """
        return
    
    async def release(self) -> None:
        """Release resources used by the embedding model (placeholder).

        Note:
            This is a placeholder implementation that does nothing.
            Concrete subclasses should override this method.
        """
        return
    
    async def reset(self) -> None:
        """Reset the embedding model's internal state (placeholder).

        Note:
            This is a placeholder implementation that does nothing.
            Concrete subclasses should override this method.
        """
        return

    async def generate(self, model_input: ModelInput) -> ModelResponse:
        """Generate a response from the embedding model (placeholder).

        Note:
            This is a placeholder implementation that returns an empty ModelResponse.
            Concrete subclasses should override this method.

        Args:
            model_input (ModelInput): Input data for the generation.

        Returns:
            ModelResponse: An empty response.
        """
        return ModelResponse()
    
    @abstractmethod
    async def batch_embedding(self, documents: List[DocumentEntry]) -> List[ModelResponse]:
        """Generate embeddings for multiple documents in batch.

        Args:
            documents (List[DocumentEntry]): List of documents to embed.

        Returns:
            List[ModelResponse]: List of embedding responses.
        """
        pass
    
class ConversationModel(BaseModel):
    """Abstract base class for conversation models.

    Conversation models handle multi-turn dialogues and provide specialized
    methods for batch responses and tool integration.

    Note:
        This class provides default implementations for most BaseModel methods,
        leaving batch_response, tool_return_integrate, and final_response_integrate
        as abstract methods that must be implemented.
    """
    def __init__(
        self, 
        base_url: str, 
        api_key: str, 
        client_configs: Dict, 
        default_generate_configs: Dict,
        instance_id: str,
        disable_openai_backend: bool = False
    ) -> None:
        """Initialize a conversation model.

        Args:
            base_url (str): Base URL for the conversation API.
            api_key (str): API key for authentication.
            client_configs (Dict): Additional client configuration.
            default_generate_configs (Dict): Default generation parameters.
            instance_id (str): Unique identifier for this instance.
            disable_openai_backend (bool, optional): Whether to disable OpenAI backend. Defaults to False.
        """
        super().__init__(
            base_url, 
            api_key, 
            client_configs, 
            default_generate_configs, 
            instance_id,
            disable_openai_backend
        )

    def clone(self) -> "ConversationModel":
        """Create a copy of this conversation model instance (placeholder).

        Note:
            This is a placeholder implementation that returns None.
            Concrete subclasses should override this method.

        Returns:
            ConversationModel: A new instance with the same configuration.
        """
        return None # type: ignore

    async def build(self) -> None:
        """Prepare the conversation model for use (placeholder).

        Note:
            This is a placeholder implementation that does nothing.
            Concrete subclasses should override this method.
        """
        return

    async def release(self) -> None:
        """Release resources used by the conversation model (placeholder).

        Note:
            This is a placeholder implementation that does nothing.
            Concrete subclasses should override this method.
        """
        return

    async def reset(self) -> None:
        """Reset the conversation model's internal state (placeholder).

        Note:
            This is a placeholder implementation that does nothing.
            Concrete subclasses should override this method.
        """
        return   

    async def generate(self, model_input: ModelInput) -> ModelResponse:
        """Generate a response from the conversation model (placeholder).

        Note:
            This is a placeholder implementation that returns an empty ModelResponse.
            Concrete subclasses should override this method.

        Args:
            model_input (ModelInput): Input data for the generation.

        Returns:
            ModelResponse: An empty response.
        """
        return ModelResponse()

    @abstractmethod
    async def batch_response(self, inputs: List[ModelInput]) -> List[ModelResponse]:
        """Generate responses for multiple inputs in batch.

        Args:
            inputs (List[ModelInput]): List of input data for batch generation.

        Returns:
            List[ModelResponse]: List of model responses.
        """
        pass

    @abstractmethod
    def tool_return_integrate(self, input: ModelInput, response: ModelResponse, tool_return: List[ToolReturn], **kwargs) -> ModelInput:
        """Integrate tool return(s) into the model input for subsequent steps.

        Args:
            input (ModelInput): Original model input.
            response (ModelResponse): Model's response that triggered tool calls.
            tool_return (List[ToolReturn]): List of tool return values.
            **kwargs: Additional keyword arguments for integration logic.

        Returns:
            ModelInput: Updated model input with tool returns incorporated.
        """
        pass

    @abstractmethod
    def final_response_integrate(self, input: ModelInput, response: ModelResponse, **kwargs) -> ModelInput:
        """Integrate the model's final response into the input for downstream processing.

        Args:
            input (ModelInput): Original model input.
            response (ModelResponse): Model's final response.
            **kwargs: Additional keyword arguments for integration logic.

        Returns:
            ModelInput: Updated model input with final response incorporated.
        """
        pass

def openai_compatiable_translate(model_input: ModelInput) -> Dict:
    """Convert a ModelInput object to an OpenAI-compatible dictionary.

    This function translates the internal ModelInput representation into a format
    compatible with OpenAI's API, including tool schemas conversion.

    Args:
        model_input (ModelInput): The model input to translate.

    Returns:
        Dict: A dictionary compatible with OpenAI's API format.

    Note:
        The function handles optional fields (model, messages, tools, input, extras)
        and converts ToolSchema objects to OpenAI function calling format.
    """
    def to_openai_function_calling_schema(tool_schema: ToolSchema) -> Dict:
        """Convert a ToolSchema to OpenAI function calling format.

        Args:
            tool_schema (ToolSchema): The tool schema to convert.

        Returns:
            Dict: OpenAI function calling schema dictionary.
        """
        properties: Dict = {}
        for param in tool_schema.params:
            param_property: Dict[str, Any] = {
                'type': param.type,
                'description': param.description
            }
            if param.enum:
                param_property['enum'] = param.enum
            properties[param.field] = param_property

        required: List = [
            param.field
            for param in tool_schema.params
            if param.required
        ]
 
        function_body: Dict = {
            'name': tool_schema.name,
            'description': tool_schema.description,
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required
            }
        }
 
        return {
            'type': 'function',
            'function': function_body
        }

    output_dict: Dict = {}
    if model_input.model is not None:
        output_dict['model'] = model_input.model
    if model_input.messages is not None:
        output_dict['messages'] = model_input.messages
    if model_input.tools is not None:
        output_dict['tools'] = [
            to_openai_function_calling_schema(schema)
            for schema in model_input.tools
        ]
    if model_input.input is not None:
        output_dict['input'] = model_input.input
    if model_input.extras is not None:
        output_dict |= model_input.extras
    return output_dict

if __name__ == '__main__':
    pass
