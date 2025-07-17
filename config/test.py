from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict
print(__file__)
class DotenvSettings(BaseSettings):
    model_config=SettingsConfigDict(env_file=Path(__file__).parent/("config.env"))
    MISTRAL_API_KEY:str
    WEATHER_API_KEY:str
    # print(model_config)

settings=DotenvSettings()
print(settings)

# print('Hello world, welcome to the world of Gen AI')