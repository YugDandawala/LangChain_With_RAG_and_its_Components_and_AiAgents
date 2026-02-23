# Taking model from firworks 
# Similar steps are there to take model from openrouter
# can alos use modle from nvdia using this way
#just change the url and api key of particular platform
from openai import OpenAI
import base64

# Vision model read from images + Textual questions also
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("Crag.png")

client = OpenAI(
    api_key = "key",
    base_url = "https://api.fireworks.ai/inference/v1"
)

response = client.chat.completions.create(
    model="accounts/fireworks/models/qwen3-vl-30b-a3b-thinking",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is Machine learning"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens = 300
)

print(response.choices[0].message.content)