import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

from src.tools.logging_tools import LOGGER
from openai.types.completion import Completion

from src.openai_utils.role import Message, MessageRole


def load_secrets(secret_path: str) -> Dict[str, str]:
    """
    Load OpenAI API credentials from a JSON file.

    Args:
        secret_path (str): Path to secrets file relative to home directory.

    Returns:
        Dict[str, str]: Dictionary containing API credentials.
    """
    secrets_file = Path.home() / secret_path
    try:
        with secrets_file.open("r") as file:
            secrets = json.load(file)
            LOGGER.debug(f"Secrets loaded from {secrets_file}")
            return secrets
    except FileNotFoundError:
        LOGGER.error(f"Secrets file not found at {secrets_file}")
        raise
    except json.JSONDecodeError:
        LOGGER.error(f"Invalid JSON format in secrets file at {secrets_file}")
        raise


class ChatConfig(BaseModel):
    """
    Configuration settings for OpenAI chat completions.
    """
    seed: int = Field(default=13, description="Random seed for reproducibility")
    model: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    max_completion_tokens: int = Field(default=1024, description="Maximum tokens in completion")
    secret_path: str = Field(default=".secrets/openai", description="Path to API credentials")
    max_retries: int = Field(default=10, description="Maximum retry attempts")
    logprobs: bool = Field(default=False, description="Whether to return log probabilities")
    response_format: Optional[Dict[str, Any]] = Field(
        default=None, description="Format for completion responses"
    )


class OpenAIChatAPI:
    """
    Client for interacting with the OpenAI chat completions API.
    """

    def __init__(self, config: ChatConfig) -> None:
        """
        Initialize the OpenAI chat API client.

        Args:
            config (ChatConfig): Configuration for the chat client.
        """
        self.config = config
        LOGGER.info(f"Initializing OpenAIChatAPI with config: {self.config}")

        secrets = load_secrets(self.config.secret_path)
        client_kwargs = self._prepare_client_kwargs(secrets)
        self.client = OpenAI(**client_kwargs)
        LOGGER.debug("OpenAI client initialized successfully.")

    def _prepare_client_kwargs(self, secrets: Dict[str, str]) -> Dict[str, str]:
        """
        Prepare keyword arguments for OpenAI client initialization.

        Args:
            secrets (Dict[str, str]): API credentials.

        Returns:
            Dict[str, str]: Keyword arguments for the OpenAI client.
        """
        kwargs = {"api_key": secrets["api_key"]}
        organization = secrets.get("organization")
        if organization:
            kwargs["organization"] = organization
            LOGGER.debug(f"Organization set to: {organization}")
        return kwargs

    def __call__(self, messages: List[Message], **kwargs: Any) -> Completion:
        """
        Generate a chat completion for the given messages.

        Args:
            messages (List[Message]): List of conversation messages.
            **kwargs: Additional arguments for the completion.

        Returns:
            Completion: The chat completion response.
        """
        message_dicts = [message.to_dict() for message in messages]
        # LOGGER.debug(f"Generating completion with messages: {message_dicts}")

        try:
            completion = self.client.with_options(
                max_retries=self.config.max_retries
            ).chat.completions.create(
                seed=self.config.seed,
                messages=message_dicts,
                model=self.config.model,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_completion_tokens,
                logprobs=self.config.logprobs,
                response_format=self.config.response_format,
                **kwargs,
            )
            # LOGGER.debug("Completion generated successfully.")
            return completion
        except Exception as e:
            LOGGER.error(f"Error generating completion: {e}")
            raise

    def parse(self, response: Completion) -> Optional[Dict[str, Any]]:
        """
        Parse the completion response.

        Args:
            response (Completion): The completion response to parse.

        Returns:
            Optional[Dict[str, Any]]: Parsed response if format is JSON, else None.
        """
        try:
            content = response.choices[0].message.content
            # LOGGER.debug(f"Parsing response content: {content}")
            if self.config.response_format == {"type": "json_object"}:
                parsed_content = json.loads(content)
                # LOGGER.debug(f"Parsed content: {parsed_content}")
                return parsed_content
            return None
        except (IndexError, KeyError, json.JSONDecodeError) as e:
            LOGGER.error(f"Error parsing response: {e}")
            return None
        

class AsyncOpenAIChatAPI:
    """
    Client for interacting with the OpenAI chat completions API.
    """

    def __init__(self, config: ChatConfig) -> None:
        """
        Initialize the OpenAI chat API client.

        Args:
            config (ChatConfig): Configuration for the chat client.
        """
        self.config = config
        LOGGER.info(f"Initializing OpenAIChatAPI with config: {self.config}")

        secrets = load_secrets(self.config.secret_path)
        client_kwargs = self._prepare_client_kwargs(secrets)
        self.client = AsyncOpenAI(**client_kwargs)
        LOGGER.debug("OpenAI client initialized successfully.")

    def _prepare_client_kwargs(self, secrets: Dict[str, str]) -> Dict[str, str]:
        """
        Prepare keyword arguments for OpenAI client initialization.

        Args:
            secrets (Dict[str, str]): API credentials.

        Returns:
            Dict[str, str]: Keyword arguments for the OpenAI client.
        """
        kwargs = {"api_key": secrets["api_key"]}
        organization = secrets.get("organization")
        if organization:
            kwargs["organization"] = organization
            LOGGER.debug(f"Organization set to: {organization}")
        return kwargs

    async def __call__(self, messages: List[Message], **kwargs: Any) -> Completion:
        """
        Generate a chat completion for the given messages.

        Args:
            messages (List[Message]): List of conversation messages.
            **kwargs: Additional arguments for the completion.

        Returns:
            Completion: The chat completion response.
        """
        message_dicts = [message.to_dict() for message in messages]
        LOGGER.debug(f"Generating completion with messages: {message_dicts}")

        try:
            completion = await self.client.with_options(
                max_retries=self.config.max_retries
            ).chat.completions.create(
                seed=self.config.seed,
                messages=message_dicts,
                model=self.config.model,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_completion_tokens,
                logprobs=self.config.logprobs,
                response_format=self.config.response_format,
                **kwargs,
            )
            LOGGER.debug("Completion generated successfully.")
            return completion
        except Exception as e:
            LOGGER.error(f"Error generating completion: {e}")
            raise

    def parse(self, response: Completion) -> Optional[Dict[str, Any]]:
        """
        Parse the completion response.

        Args:
            response (Completion): The completion response to parse.

        Returns:
            Optional[Dict[str, Any]]: Parsed response if format is JSON, else None.
        """
        try:
            content = response.choices[0].message.content
            LOGGER.debug(f"Parsing response content: {content}")
            if self.config.response_format == {"type": "json_object"}:
                parsed_content = json.loads(content)
                LOGGER.debug(f"Parsed content: {parsed_content}")
                return parsed_content
            return None
        except (IndexError, KeyError, json.JSONDecodeError) as e:
            LOGGER.error(f"Error parsing response: {e}")
            return None


if __name__ == "__main__":
    chat_config = ChatConfig(response_format={"type": "json_object"})
    chat_api = OpenAIChatAPI(chat_config)
