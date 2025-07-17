
from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict,PydanticBaseSettingsSource,TomlConfigSettingsSource
from pydantic import BaseModel
from typing import Tuple,Type


class LLM(BaseModel):
    name:str
    base_url:str
    api_key:str
    max_tokens:int
    temperature:float
    prompt:str

class Logfire(BaseModel):
    token:str




class TomlSetting(BaseSettings):

    model_config=SettingsConfigDict(toml_file=Path(__file__).resolve().parent/"config.toml")
    # print(model_config)

    llm:LLM
    logfire:Logfire

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # print("Print inside class method",settings_cls.model_config['toml_file'])
        # print(settings_cls.model_fields['llm'])
        return (TomlConfigSettingsSource(settings_cls),)

settings=TomlSetting()

if __name__=='__main__':
    print(settings)
    print(settings.model_dump())