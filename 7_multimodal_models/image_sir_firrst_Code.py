from typing import List
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
import os
import base64
from textwrap import dedent
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


class ImageLoaderBase64:

    def __init__(self, user_prompt: str, image_file_path: str):
        with open(image_file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        self.encoded_message_with_image = [
            {"type": "text", "text": f"{user_prompt}"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high",
                },
            },
        ]


system_prompt = """
    You are smart vision agent with text extraction skills
"""

agent = Agent(
    model=OpenAIModel(
        model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    ),
    system_prompt=dedent(system_prompt),
)

image_request_1 = ImageLoaderBase64(
    user_prompt="extract all text",
    image_file_path=os.path.expanduser(
        "~/Downloads/All-I-want-to-know-is-where-i-am-going-to-die.png"
    ),
)
result = agent.run_sync(image_request_1.encoded_message_with_image)
print(result.data)