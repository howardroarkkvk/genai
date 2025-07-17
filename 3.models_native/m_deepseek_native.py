from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)
# print(os.getenv("DEEPSEEK_API_KEY"))

client=OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url="https://api.deepseek.com")
#print(client)

response=client.chat.completions.create(model=os.getenv("DEEPSEEK_CHAT_MODEL"),
                               messages=[{"role":"system","content":"You are a helpful assistant"},
                                         {"role":"user","content":"Tell me about Charlie Munger speeches"}] ,stream=False )


print(response.choices[0].message.content)