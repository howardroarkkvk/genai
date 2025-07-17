from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/novita/v3/openai",
    api_key=os.getenv('HF_TOKEN'),
)

completion = client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    # max_tokens=512,
)

print(completion.choices[0].message)