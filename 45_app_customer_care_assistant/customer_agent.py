from pydantic_ai import Agent,RunContext
from customer_db_service import *
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Annotated
from llm import * 

@dataclass
class CustomerDependencies:
    customer_id:int
    db:CustomerDB


class CustomerResult(BaseModel):
    support_adivce:Annotated[str,...,'Advice returned to the customer']
    block_card:Annotated[bool,...,'Whether to block their card or not']
    risk:Annotated[int,...,'Risk level of query']
    customer_tracking_id:Annotated[str,...,'Tracking ID of customer']
                                   
system_prompt=(
        "You are a support agent in our bank, give the "
        "customer support and judge the risk level of their query. "
    )

customer_agent=Agent(name='CustomerAgent',model=build_model(),system_prompt=system_prompt,output_type=CustomerResult,deps_type=CustomerDependencies,retries=3)

@customer_agent.tool
async def block_card(ctx:RunContext[CustomerDependencies]):
    return f"i m sorry to hear that ,{ctx.deps.db.customer_name(ctx.deps.customer_id)},we are temporarily blocking your card to prevent unauthorized acceess"

@customer_agent.tool
async def customer_balance(ctx:RunContext[CustomerDependencies],include_pending:str):
    """Returns the customers current account balance"""
    balance = await ctx.deps.db.customer_balance(id=ctx.deps.customer_id,include_pending=include_pending)
    return f'${balance:.2f}'

