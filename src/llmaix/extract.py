import inspect
import json
import os
from typing import Any, Type, TypeVar

import openai
import pydantic
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


def convert_messages_to_types(
    messages: list[dict[str, str]],
) -> list[
    ChatCompletionAssistantMessageParam
    | ChatCompletionUserMessageParam
    | ChatCompletionSystemMessageParam
]:
    """Converts a list of message dictionaries to OpenAI message types.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.

    Returns:
        List of OpenAI message types.
    """
    converted_messages = []
    for message in messages:
        if message["role"] == "user":
            converted_messages.append(ChatCompletionUserMessageParam(**message))
        elif message["role"] == "assistant":
            converted_messages.append(ChatCompletionAssistantMessageParam(**message))
        elif message["role"] == "system":
            converted_messages.append(ChatCompletionSystemMessageParam(**message))
        else:
            raise ValueError(
                f"Invalid role '{message['role']}' in message. "
                "Valid roles are 'user', 'assistant', and 'system'."
            )
    return converted_messages


def make_llm_request(
    client: openai.OpenAI,
    llm_model: str,
    prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    api_type: str = "chat/completions",
    json_schema: str | dict[str, Any] | list[Any] | None = None,
    pydantic_model: Type[PydanticModel] | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
) -> ChatCompletion:
    """Makes a request to the LLM API using the provided parameters.

    Handles both chat/completions and responses API types, supporting structured
    output through JSON schema or Pydantic models.

    Args:
        client: OpenAI client instance used to make API calls
        llm_model: Model identifier string (e.g., "gpt-4", "claude-3-opus-20240229")
        prompt: Optional text prompt (used with responses API)
        messages: List of message objects for chat/completions API
        api_type: API endpoint type, either "chat/completions" or "responses"
        json_schema: Optional JSON schema to structure the model's response
        pydantic_model: Optional Pydantic model class to structure and validate the response
        temperature: Optional sampling temperature for controlling randomness
        max_completion_tokens: Optional maximum token limit for the response

    Returns:
        str: The text content from the model's response

    Raises:
        ValueError: If required parameters are missing or incompatible
        NotImplementedError: If the specified API type is not implemented
    """

    if api_type == "chat/completions":
        if messages:
            params: dict[str, Any] = {
                "model": llm_model,
                "messages": convert_messages_to_types(messages),
            }
            if temperature:
                params["temperature"] = temperature

            if max_completion_tokens:
                params["max_completion_tokens"] = max_completion_tokens

            if pydantic_model:
                params["response_format"] = pydantic_model
                completion = client.beta.chat.completions.parse(**params)
                return completion
            elif json_schema:
                if isinstance(json_schema, str):
                    json_schema = json.loads(json_schema)
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "strict": True,
                        "schema": json_schema,
                    },
                }
            return client.chat.completions.create(**params)
        else:
            raise ValueError(
                "You must provide messages if you are using the chat/completions API endpoint."
            )

    elif api_type == "responses":
        raise NotImplementedError(
            "The responses API endpoint is not implemented yet. TODO"
        )
    else:
        raise NotImplementedError(f"API type {api_type} not implemented.")


def extract_info(
    llm_model: str = "",
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    client: openai.OpenAI | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_type: str = "chat/completions",
    json_schema: str | dict[str, Any] | list[Any] | None = None,
    pydantic_model: Type[PydanticModel] | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    include_full_completion_result: bool = False,
) -> str | ChatCompletion | dict[str, Any] | list[Any]:
    """Extracts information from text using an LLM with flexible configuration options.

    This function provides a convenient interface for querying LLMs with various
    input formats and configuration options. It handles API authentication,
    message formatting, and structured output with JSON schema or Pydantic models.

    Args:
        llm_model: Model identifier string (e.g., "gpt-4") - required unless set in environment
        prompt: Text prompt to send to the model (cannot be used with messages)
        system_prompt: Optional system message to set context (used with prompt)
        messages: List of message dictionaries with role and content (cannot be used with prompt)
        client: Optional pre-configured OpenAI client instance
        base_url: Optional API base URL for non-OpenAI endpoints
        api_key: Optional API key (required if client not provided)
        api_type: API endpoint type, either "chat/completions" or "responses"
        json_schema: Optional JSON schema to structure the model's response
        pydantic_model: Optional Pydantic model class to structure and validate the response
        temperature: Optional sampling temperature for controlling randomness
        max_completion_tokens: Optional maximum token limit for the response
        include_full_completion_result: If True, returns the full completion result instead of just the text

    Returns:
        str: The text content from the model's response if no json_schema or pydantic_model is provided
        openai.types.chat.ChatCompletion: The full completion result if include_full_completion_result is True
        dict[str, Any] or list[Any]: The structured response if json_schema or pydantic_model is provided

    Raises:
        ValueError: If required parameters are missing or incompatible
        TypeError: If pydantic_model is not a valid Pydantic model class
        ValueError: If the API request fails
    """

    if api_type not in ["chat/completions", "responses"]:
        raise ValueError("api_type must be either chat/completions or responses.")

    if api_type == "responses":
        if system_prompt:
            raise ValueError(
                "You cannot provide a system_prompt if you are using the responses API endpoint."
            )
        if not prompt:
            raise ValueError(
                "You must provide a prompt if you are using the responses API endpoint."
            )
        if messages:
            raise ValueError(
                "You cannot provide messages if you are using the responses API endpoint."
            )

    if json_schema and pydantic_model:
        raise ValueError(
            "You can only provide one of json_schema or pydantic_model, not both."
        )

    if not client:
        # check if api_key is provided
        if not api_key:
            if os.environ.get("OPENAI_API_KEY"):
                api_key = os.environ.get("OPENAI_API_KEY")
            else:
                raise ValueError("API key is required if client is not provided.")
        if not base_url:
            if os.environ.get("OPENAI_API_BASE"):
                base_url = os.environ.get("OPENAI_API_BASE")
        if base_url:
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
        else:
            client = openai.OpenAI(api_key=api_key)

    if not llm_model:
        if os.environ.get("OPENAI_MODEL"):
            llm_model = os.environ.get("OPENAI_MODEL", "")
        if not llm_model:
            raise ValueError("llm_model is required")

    # check if either prompt or messages is provided but not both
    if not prompt and not messages:
        raise ValueError("Either prompt or messages must be provided.")
    if prompt and messages:
        raise ValueError("You can only provide one of prompt or messages, not both.")
    if system_prompt and not prompt:
        raise ValueError("You must provide a prompt if you provide a system_prompt.")
    if system_prompt and messages:
        raise ValueError(
            "You can only provide one of system_prompt or messages, not both."
        )

    if json_schema:
        # check if json_schema is valid
        if isinstance(json_schema, str):
            try:
                json.loads(json_schema)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON schema: {e}")
        elif isinstance(json_schema, dict) or isinstance(json_schema, list):
            try:
                json.loads(json.dumps(json_schema))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON schema: {e}")

    elif pydantic_model:
        if not inspect.isclass(pydantic_model):
            raise TypeError(
                f"Expected a Pydantic model class, got an instance: {pydantic_model}"
            )
        if not issubclass(pydantic_model, pydantic.BaseModel):
            raise TypeError(
                f"Expected a Pydantic model class, got an instance: {pydantic_model}"
            )

    if api_type == "chat/completions":
        if messages:
            # check if messages is valid
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("messages must be a list of dictionaries.")
                if "role" not in message or "content" not in message:
                    raise ValueError("Each message must have a role and content.")
                if not isinstance(message["content"], str):
                    raise ValueError("Content must be a string.")
                if not isinstance(message["role"], str):
                    raise ValueError("Role must be a string.")
        else:
            # check if prompt is valid
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string.")
            if system_prompt and not isinstance(system_prompt, str):
                raise ValueError("system_prompt must be a string.")
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        # TODO: Also use stats
        completion = make_llm_request(
            client,
            llm_model,
            prompt,
            messages,
            api_type,
            json_schema,
            pydantic_model,
            temperature,
            max_completion_tokens,
        )

        if include_full_completion_result:
            return completion
        else:
            if pydantic_model or json_schema:
                if completion.choices[0].finish_reason == "length":
                    raise ValueError(
                        "The model's response was too long and was truncated. "
                        "Please increase the max_completion_tokens parameter."
                    )
                if not completion.choices or not completion.choices[0].message.content:
                    raise ValueError(
                        "The model did not return any content. "
                        "Please check your prompt, messages, or model settings."
                    )
                try:
                    return json.loads(completion.choices[0].message.content)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON response from the model: {e}. "
                        "Please check the json_schema or pydantic_model or your model's settings."
                    )
            else:
                return completion.choices[0].message.content
    except openai.OpenAIError as e:
        raise ValueError(f"Error making request to LLM API: {e}")
