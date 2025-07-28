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
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one word."},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
                ],
            }
        ],
        "max_tokens": 3,
    }
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=10)
        data = response.json()
        if not response.ok or "choices" not in data:
            return False
        reply = data["choices"][0].get("message", {}).get("content", "").lower()
        # You may want to adjust the check below based on expected behavior
        if (
            "white" in reply
            or "blank" in reply
            or "cannot" in reply
            or "empty" in reply
        ):
            return True
        return False
    except Exception:
        return False
