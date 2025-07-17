# create a free API key at https://www.tomorrow.io/weather-api/
import time
import logfire
from dotenv import load_dotenv
from httpx import Client
from dataclasses import dataclass
from pydantic import BaseModel
import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent,RunContext


@dataclass
class WeatherDeps:
    url:str
    client:Client
    weather_api_key:str

class WeatherInfo(BaseModel):
    temparature:float
    description:str

model=OpenAIModel(model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
system_prompt="You are helpful assistant. You must use the tool 'get_weather' to find the weather at given coordinates."
weather_agent= Agent(model=model,system_prompt=system_prompt,result_type=WeatherInfo,retries=1)

@weather_agent.tool
def get_weather(ctx:RunContext[WeatherDeps],lat:float,lng:float)->dict[str,any]:
    print("In get_weather")
    """Get the weather for the given lat and lng co-ordinates"""
    client1=ctx.deps.client
    loc=f'{lat},{lng}'
    print(loc)
    r=client1.get(url=ctx.deps.url,params={'location':loc,'apikey':ctx.deps.weather_api_key})#,'units':'metric'
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
    # data.get('timelines').get('minutely')[0]['values']['temperature']

if __name__=='__main__':
    load_dotenv(override=True)
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
    time.sleep(1)
    logfire.instrument_openai()
    logfire.instrument_httpx()
    print(os.getenv('WEATHER_API_KEY'))

    weatherdeps=WeatherDeps(url='https://api.tomorrow.io/v4/weather/forecast',client=Client(),weather_api_key=os.getenv('WEATHER_API_KEY'))
    resp=weather_agent.run_sync('find the weather at the following co-ordinates: at =13.0836939  ,lng= 80.270186',deps=weatherdeps)# hyd : lat =17.385044 ,lng= 78.486671
    print(resp.data)











