import logfire
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# allows you to setup and initilize your logging configuration for the application
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))

# instruments are like sensors, it collects the log information,metrics like requests and memory handling and traces --> how the journey of a requesst with in the program is traversing
logfire.instrument_openai() # 


client=OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response=client.chat.completions.create(model=os.getenv('OPENAI_CHAT_MODEL'),
                               messages=[{'role':'system','content':'You are a helpful assistant'},
                                         {'role':'user','content':'what is value investing'}])


print(response.choices[0].message.content)

