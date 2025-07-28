import json

from dotenv import load_dotenv

from llmaix.extract import extract_info


def test_extract_info():
    load_dotenv()

    extracted_text = extract_info(
        prompt="What is the capital of France? Answer in one word without punctation!"
    )

    assert extracted_text == "Paris"


def test_extract_info_with_system_prompt():
    load_dotenv()

    extracted_text = extract_info(
        prompt="What is the capital of France?",
        system_prompt="You are a helpful assistant. Answer in one word without punctation!",
    )

    assert extracted_text == "Paris"


def test_extract_info_with_messages():
    load_dotenv()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer in one word without punctation!",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ]

    extracted_text = extract_info(messages=messages)

    assert extracted_text == "Paris"


def test_extract_info_with_system_prompt_prompt_and_messages():
    load_dotenv()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer in one word without punctation!",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ]

    try:
        extract_info(
            prompt="What is the capital of France?",
            system_prompt="You are a helpful assistant. Answer in one word without punctation!",
            messages=messages,
        )
        assert False, "Expected ValueError not raised"
    except ValueError as e:
        assert str(e) == "You can only provide one of prompt or messages, not both."


def test_extract_info_with_json_schema_str():
    load_dotenv()

    json_schema = {
        "type": "object",
        "properties": {"capital": {"type": "string"}},
        "required": ["capital"],
    }

    json_schema_str = json.dumps(json_schema)

    extracted_json = extract_info(
        prompt="What is the capital of France?", json_schema=json_schema_str
    )

    assert isinstance(extracted_json, dict) or isinstance(extracted_json, list)
    assert extracted_json == {"capital": "Paris"}


def test_extract_info_with_json_schema():
    load_dotenv()

    json_schema = {
        "type": "object",
        "properties": {"capital": {"type": "string"}},
        "required": ["capital"],
    }

    extracted_json = extract_info(
        prompt="What is the capital of France?", json_schema=json_schema
    )

    assert isinstance(extracted_json, dict) or isinstance(extracted_json, list)
    assert extracted_json == {"capital": "Paris"}


def test_extract_info_with_pydantic_model():
    load_dotenv()

    from pydantic import BaseModel

    class Capital(BaseModel):
        capital: str

    extracted_json = extract_info(
        prompt="What is the capital of France?", pydantic_model=Capital
    )

    assert isinstance(extracted_json, dict) or isinstance(extracted_json, list)
    assert extracted_json == {"capital": "Paris"}
