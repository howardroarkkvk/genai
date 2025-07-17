from dataclasses import dataclass
from pydantic_ai import Agent,RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel
from httpx import Client
import os
import logfire
from dotenv import load_dotenv

# In this progream the important concepts to learn are dataclass, which helps to use the @dataclass decorator to skip writing th init() methods for a class
# Also ,instead of giving decorator to the function as @agent.run_plain, we can pass the function name in the agent in tools parameter
# and defining RunContext which is pydantic_ai package to create a context which is used for dependecny injection to pass the parameters during the run time to the agent.
# also once we create the object of the class (dataclass type), we have to add the object to the deps parameter of the user prompt


load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


model=OpenAIModel(model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class GeoCodeDeps:
    url:str
    client:Client
    api_key:str




def get_lat_lang(rt:RunContext[GeoCodeDeps],location:str)->str:
    print("location captured is :",location)
    print(rt.deps.url)
    print(rt.deps.api_key)
    r=rt.deps.client.get(rt.deps.url,params={'q':location,'api_key':rt.deps.api_key})
    # print(r)
    # print(r.json())
    return r.text#str(r.json())


deps=GeoCodeDeps(url='https://geocode.maps.co/search',client=Client(),api_key=os.getenv('GEO_API_KEY'))

system_prompt="""    Be concise, reply with one sentence.
    Use the `get_lat_lng` tool to get the latitude and longitude of a location.
"""
    
agent = Agent(model=model,system_prompt=system_prompt,tools=[get_lat_lang])



result = agent.run_sync("Find the longitude and lattitude  of delhi",deps=deps)
print(result.data)