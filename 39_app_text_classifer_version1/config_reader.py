from typing import Tuple, Type
from pydantic import BaseModel
from pathlib import Path
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class ChunkerTomlSettings(BaseModel):
    chunk_size: int
    chunk_overlap: int


class LLMTomlSettings(BaseModel):
    name:str
    base_url:str
    api_key:str
    max_tokens:int
    temperature:float
    prompt:str


class FilePathsTomlSettings(BaseModel):
    src_dir: str
    eval_file: str
    eval_file_with_response: str
    results_dir: str
    results_report_file:str
    results_matrix_file:str
    # results_non_llm_metrics_file: str


class DocSizeTomlSettings(BaseModel):
    short_doc_threshold: int
    medium_doc_threshold: int
    long_doc_threshold: int


class EmbedderTomlSettings(BaseModel):
    model:str
    token:str
    normalize:bool


class LogfireTomlSettings(BaseModel):
    token: str


class TomlSettings(BaseSettings):
    model_config = SettingsConfigDict(toml_file=Path(__file__).parent / "config.toml")

    llm: LLMTomlSettings
    file_paths: FilePathsTomlSettings
    # docs: DocSizeTomlSettings
    # chunker: ChunkerTomlSettings
    logfire: LogfireTomlSettings
    # judge_llm: LLMTomlSettings
    embedder: EmbedderTomlSettings

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


settings = TomlSettings()

if __name__ == "__main__":
    print(settings)