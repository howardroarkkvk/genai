from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv(override=True)

client=Mistral(api_key=os.getenv('MISTRAL_API_KEY'))

response=client.chat.complete(model=os.getenv("MISTRAL_CHAT_MODEL"),messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"details of Charlie mungers speeches "}
])

print(response.choices[0].message.content)