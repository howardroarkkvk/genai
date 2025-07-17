

from openai import OpenAI
import os
import logfire
from dotenv import load_dotenv
import time
import logfire



openai_client_instance=OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()


def running_openai_model(query):
    completions=openai_client_instance.chat.completions.create(model=os.getenv('OPENAI_CHAT_MODEL'),messages=[{'role':'system','content':'You are a Helpful Assistant'},
                                                                                              {'role':'user','content':query}])
    output=completions.choices[0].message.content

    return output

def driver():
    for _ in range(3):
        print(time.asctime())
        res=running_openai_model('What is the capital of india')
        print(res)


st=time.perf_counter()
driver()
et=time.perf_counter()
print(f' The elapsed time is {et-st} in seconds')





