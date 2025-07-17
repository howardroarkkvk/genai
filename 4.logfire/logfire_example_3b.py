import logfire
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))

client=genai.Client(api_key=os.getenv('GEMINI_API_KEY'))



def chat_with_gemini(query):
    response=client.models.generate_content(model=os.getenv('GEMINI_CHAT_MODEL'),contents=query)
    return response.text

with logfire.span("calling gemini model") as span:
    print(chat_with_gemini('what is value investing'))
    print(chat_with_gemini('who is victor frankel'))
    print(chat_with_gemini('who is charlie munger'))
