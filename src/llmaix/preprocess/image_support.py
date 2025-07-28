import base64
import io

import requests
from PIL import Image


def test_remote_image_support(api_url: str, model: str, api_key: str) -> bool:
    img = Image.new("RGB", (1, 1), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Describe this image in one word."},
            {"role": "user", "name": "image", "image": b64},
        ],
        "max_tokens": 3,
    }
    response = requests.post(api_url, json=payload, headers=headers, timeout=10)
    try:
        data = response.json()
    except Exception:
        return False
    return response.ok and "choices" in data
