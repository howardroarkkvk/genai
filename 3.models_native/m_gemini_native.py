from google import genai
from dotenv import load_dotenv
import os

# this loads the gemini api key 
load_dotenv(override=True)
# print(os.getenv("GEMINI_API_KEY"))
# print(os.getenv("GEMINI_API_KEY"))


client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

#client.chats.completions.create()
response=client.models.generate_content(model=os.getenv("GEMINI_CHAT_MODEL"),contents="what is recursion in programming")

print(response.text)
