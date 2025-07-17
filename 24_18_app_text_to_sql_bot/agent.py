
from utils import get_schema_info
from dataclasses import dataclass
from pydantic_ai import Agent,RunContext
from llm import build_from_model_name

@dataclass
class Deps:
    db_path:str


def get_agent(model_name:str,system_prompt:str):

    model=build_from_model_name(model_name=model_name)
    sql_agent=Agent(model=model,system_prompt=system_prompt,deps_type=Deps)

    @sql_agent.system_prompt
    def add_context_system_prompt(ctx:RunContext[Deps])->str:
        schema_text=get_schema_info(ctx.deps.db_path)
        return '\n<schema>\n' + schema_text + '\n</schema>'
    
    return sql_agent





