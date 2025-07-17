from loan_db_service import *
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Annotated
from pydantic_ai import Agent,RunContext
from llm import *
@dataclass

class LoanDependencies:
    customer_id:int
    db:LoanDB


class LoanResult(BaseModel):
    loan_approval_status: Annotated[str,...,'Approval status of loan (e.g Approved, Denied, Pending)']
    loan_balance: Annotated[float,...,'Remaining balance of the loan']

system_prompt=(
        "You are a support agent in our bank, assisting customers with loan-related inquiries. "
        "For every query, provide the following information: "
        "- Loan approval status (e.g., Approved, Denied, Pending) "
        "- Loan balance "
        "Please ensure that your response is clear and helpful for the customer. "
        "Always conclude by providing the customer’s name and capturing their information in the marking system using "
        "the tool `capture_customer_name`. "
        "Never generate data based on your internal knowledge; always rely on the provided tools to fetch the most "
)

loan_agent=Agent(name='LoanAgent',model=build_model(),system_prompt=system_prompt,deps_type=LoanDependencies,output_type=LoanResult,retries=3)


@loan_agent.tool
async def loan_status(ctx:RunContext[LoanDependencies])->str:
    print('Loan Status tool call')
    status = await ctx.deps.db.loan_status(id=ctx.deps.customer_id)
    return f"the loan status is {status!r}"

@loan_agent.tool
async def cancel_loan(ctx:RunContext[LoanDependencies])->str:
    return await ctx.deps.db.cancel_loan(id=ctx.deps.customer_id)

@loan_agent.tool
async def add_loan(ctx:RunContext[LoanDependencies],amount:float,interest_rate:float):
    return await ctx.deps.db.add_loan(id=ctx.deps.customer_id,amount=amount,interest_rate=interest_rate)

@loan_agent.tool
async def loan_balance(ctx:RunContext[LoanDependencies]):
    return await ctx.deps.db.loan_balance(id=ctx.deps.customer_id)



