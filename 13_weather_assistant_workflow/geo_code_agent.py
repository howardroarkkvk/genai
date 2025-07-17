# create a free API key at https://geocode.maps.co/
from pydantic import BaseModel
from httpx import Client
from dataclasses import dataclass
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent,RunContext
import os
from dotenv import load_dotenv
import logfire
import time

# these are used for passing the values to the agent function 
@dataclass
class GeoCodeDeps:
    url:str
    client:Client
    geocode_api_key:str

class GeoCoOrdinates(BaseModel):
    lat:float
    long:float

model=OpenAIModel(model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
system_prompt="""You are helpful assistant , you must use the tool 'get_lat_lng' to find the latitude and longitude of the given location """
geo_agent=Agent(model=model,system_prompt=system_prompt,result_type=GeoCoOrdinates)

@geo_agent.tool
def get_lat_lng(ctx:RunContext[GeoCodeDeps],location:str)->str:
    print("In get_lat_lng")
    try:
        client=ctx.deps.client
        r=client.get(url=ctx.deps.url,params={'q':location,'api_key':ctx.deps.geocode_api_key})
        print("in get lat lng",r.json())
        r.raise_for_status()
    except Exception as E:
        print(e)
        return {"lat": 17.385044, "lng": 78.486671}
    return r.json()

if __name__=='__main__':
    load_dotenv(override=True)
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
    time.sleep(1)
    logfire.instrument_openai()
    logfire.instrument_httpx()
    geodeps=GeoCodeDeps(
        url="https://geocode.maps.co/search",
        client=Client(),
        geocode_api_key=os.getenv("GEO_API_KEY"),)
    result=geo_agent.run_sync("Find chennai's latitude and longitude",deps=geodeps)
    print(result.data)





