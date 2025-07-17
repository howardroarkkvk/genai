import logfire
import os
from dotenv import load_dotenv
from google import genai


load_dotenv(override=True)

# allows you to setup and initilize your logging configuration for the application for logfire
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))

# this establishes the client connection with gemini model
client=genai.Client(api_key=os.environ['GEMINI_API_KEY'])


# applying instrument to a function --> instrument basically acts like sensor collecting logs, timings and traces of the requests
@logfire.instrument("Calling Gemini Model",span_name='Span 1',extract_args=True)
def chat_with_gemini(query):
    response=client.models.generate_content(model=os.environ['GEMINI_CHAT_MODEL'],contents=query)
    return response.text


print(chat_with_gemini('who is victor frankl'))



