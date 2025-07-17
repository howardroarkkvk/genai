from typing import Tuple,Type
from pydantic import BaseModel
from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic_settings import PydanticBaseSettingsSource,TomlConfigSettingsSource



config_path=Path(__file__).parent/"config.toml"
# print(config_path)



class LLMTomlSettings(BaseModel):
    name:str
    base_url:str
    api_key:str
    max_tokens:float
    temperature:float
    prompt:str

class EmbedderTomlSettings(BaseModel):
    model:str
    token:str
    normalize:bool

class ChunkerTomlSettings(BaseModel):
    chunk_strategy:str


class RetrieverTomlSettings(BaseModel):
    top_k:int


class RerankerTomlSettings(BaseModel):
    model:str
    token:str
    top_k:int

class FilepathsTomlSettings(BaseModel):
    src_dir:str
    db_file:str
    kb_dir:str
    vector_db_dir:str
    eval_file:str
    eval_file_with_response:str
    results_dir:str

class LogfireTomlSettings(BaseModel):
    token:str

class JudgellmTomlSettings(BaseModel):
    name:str
    base_url:str
    api_key:str
    max_tokens:int
    temparature:float
    prompt:str


class TomlSettings(BaseSettings):
    model_config=SettingsConfigDict(toml_file=Path(__file__).parent/"config.toml")

    llm:LLMTomlSettings
    embedder:EmbedderTomlSettings
    chunker:ChunkerTomlSettings
    retriever:RetrieverTomlSettings
    reranker:RerankerTomlSettings
    file_paths:FilepathsTomlSettings
    logfire:LogfireTomlSettings
    judge_llm:JudgellmTomlSettings


    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)

settings=TomlSettings()
# print(settings)
# print(settings.llm.api_key)


