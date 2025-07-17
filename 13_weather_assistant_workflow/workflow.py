from dotenv import load_dotenv
import os
import logfire


from geo_code_agent import *
from weather_agent import *



def workflow_chain(userprompt:str,geodeps:GeoCodeDeps,weatherdeps:WeatherDeps):
    print("before running geo_result")
    geo_result=geo_agent.run_sync(user_prompt=userprompt,deps=geodeps)
    coordinates=geo_result.data

    print("before running weather_result")

    weather_result=weather_agent.run_sync(f"find the weather at the following co ordinates: {coordinates}",deps=weatherdeps)
    return weather_result.data


def main(userprompt:str):
    load_dotenv(override=True)
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
    time.sleep(1)
    logfire.instrument_openai()
    logfire.instrument_httpx()

    client=Client()
    geodeps=GeoCodeDeps(url="https://geocode.maps.co/search",client=Client(),geocode_api_key=os.getenv("GEO_API_KEY"))
    weatherdeps=WeatherDeps(url='https://api.tomorrow.io/v4/weather/forecast',client=Client(),weather_api_key=os.getenv('WEATHER_API_KEY'))
    result=workflow_chain(userprompt,geodeps,weatherdeps)
    return result




if __name__=="__main__":
    userprompt="Find Coimbatore's current weather"
    response=main(userprompt)
    print(response)
