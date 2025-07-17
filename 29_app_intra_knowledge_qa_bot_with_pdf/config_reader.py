
from typing import Tuple,Type
from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict,PydanticBaseSettingsSource,TomlConfigSettingsSource
from pydantic import BaseModel

class llm(BaseModel):
    name:str
    base_url:str
    api_key:str
    max_tokens:int
    temperature:float
    prompt:str

class embedder(BaseModel):

    model:str
    token:str
    normalize:bool

class chunker(BaseModel):
    chunk_size:int
    chunk_overlap:int


class retriever(BaseModel):
    top_k:int

class reranker(BaseModel):

    model:str
    token:str
    top_k:int

class file_paths(BaseModel):
    src_dir:str
    kb_dir:str
    vector_db_dir:str
    bm_25_dir:str
    eval_file:str
    eval_file_with_response:str
    results_dir:str

class logfire(BaseModel):
    token:str

class judge_llm(BaseModel):

    name:str
    base_url:str
    api_key:str
    max_tokens:int
    temparature:float
    prompt:str







class TomlSettings(BaseSettings):
    model_config=SettingsConfigDict(toml_file=Path(__file__).resolve().parent/'config.toml')

    llm:llm
    embedder:embedder
    chunker:chunker
    retriever:retriever
    reranker:reranker
    file_paths:file_paths
    logfire:logfire
    judge_llm:judge_llm

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

if __name__=='__main__':
    settings=TomlSettings()
    print(settings)
    # print(settings.llm.max_tokens)
    # print(settings.llm)