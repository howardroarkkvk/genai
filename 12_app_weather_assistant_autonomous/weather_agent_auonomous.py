import time
import os
from pydantic_ai import Agent,RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.mistral import MistralModel
from pydantic import BaseModel
from httpx import Client
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire

@dataclass
class Geo_Weather_deps:
    client:Client
    geo_url:str
    weather_url:str
    geo_api_key:str
    weather_api_key:str


class WeatherOutput(BaseModel):
    temparature:float
    description:str

system_prompt="You are a reasoning assistanct, Please use the following , please use the following functions 'get_lat_lng', 'get_weather_update' to get the temparate of the city"
model=OpenAIModel(model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
# model=MistralModel(model_name=os.environ['MISTRAL_CHAT_MODEL'],api_key=os.getenv('MISTRAL_API_KEY'))
# model=OpenAIModel(model_name=os.getenv("OPENAI_REASONING_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
# model=OpenAIModel(model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY"))
geo_weather_agent=Agent(model=model,system_prompt=system_prompt,result_type=WeatherOutput,retries=1,deps_type=Geo_Weather_deps)

@geo_weather_agent.tool
def get_lat_lng(ctx:RunContext[Geo_Weather_deps],location:str)->str:
    client=ctx.deps.client
    r=client.get(url=ctx.deps.geo_url,params={'q':location,'api_key':ctx.deps.geo_api_key})
    data=r.json()
    print(data)
    return data

@geo_weather_agent.tool
def get_weather_update(ctx:RunContext[Geo_Weather_deps],lat:str,lng:str)->str:
    client1=ctx.deps.client
    loc=f'{lat},{lng}'
    print(loc)
    r=client1.get(url=ctx.deps.weather_url,params={'location':loc,'apikey':ctx.deps.weather_api_key})#,'units':'metric'
    r.raise_for_status()
    data=r.json()
    print('in get weather',data.get('timelines').get('minutely')[0]['values']['temperature'])
    code_lookup = {
        1000: "Clear, Sunny",
        1100: "Mostly Clear",
        1101: "Partly Cloudy",
        1102: "Mostly Cloudy",
        1001: "Cloudy",
        2000: "Fog",
        2100: "Light Fog",
        4000: "Drizzle",
        4001: "Rain",
        4200: "Light Rain",
        4201: "Heavy Rain",
        5000: "Snow",
        5001: "Flurries",
        5100: "Light Snow",
        5101: "Heavy Snow",
        6000: "Freezing Drizzle",
        6001: "Freezing Rain",
        6200: "Light Freezing Rain",
        6201: "Heavy Freezing Rain",
        7000: "Ice Pellets",
        7101: "Heavy Ice Pellets",
        7102: "Light Ice Pellets",
        8000: "Thunderstorm",
    }
    return {"temperature":f'{data.get('timelines').get('minutely')[0]['values']['temperature']:0.0f}°C',
    "description":code_lookup.get(data.get('timelines').get('minutely')[0]['values']['weatherCode'])}

if __name__=='__main__':
    load_dotenv(override=True)
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
    time.sleep(1)
    logfire.instrument_openai()
    logfire.instrument_httpx()
    geodeps=Geo_Weather_deps(client=Client(),
                             geo_url='https://geocode.maps.co/search',
                             weather_url='https://api.tomorrow.io/v4/weather/forecast',
                             geo_api_key=os.getenv('GEO_API_KEY'),
                             weather_api_key=os.getenv('WEATHER_API_KEY'))
    
    resp=geo_weather_agent.run_sync("Find the Hyderabad's weather",deps=geodeps)
    print(resp.data)

