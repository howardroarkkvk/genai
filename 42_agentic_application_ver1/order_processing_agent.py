from pydantic_ai import RunContext,Agent
from dataclasses import dataclass
from order_db_service import *
from llm import *
from config_reader import *
from typing import Annotated
from pydantic import BaseModel
from uuid import uuid4

# This program has all the deps, functions and also an ai agent....(with functions defined)

@dataclass
class Deps:
    order_db_service:OrderDBService

def get_order_details(ctx:RunContext[Deps],order_id:str):
    """Get the current status and details of an order"""
    order=ctx.deps.order_db_service.get_order(order_id)
    if not order:
        return "Order not found"
    # print(order.model_dump_json())
    return order.model_dump_json()


def update_shipping_address(ctx:RunContext[Deps],order_id:str,street:str,city:str, postal_code:str,country:str):
    """update the shipping address for an order if possible"""
    order=ctx.deps.order_db_service.get_order(order_id)
    if not order:
        return 'Order not found'
    if not order.can_modify:
        return f'cannot modify the order, current status is {order.status.value}'
    
    new_address=Address(street=street,city=city,postal_code=postal_code,country=country)
    if ctx.deps.order_db_service.update_shipping_address(order_id=order_id,new_address=new_address):
        return f'successfully updated the shipping address'
    return f'Failed to update shipping address'

def cancel_order(ctx:RunContext[Deps],order_id:str):
    """cancel an order if possible"""
    order=ctx.deps.order_db_service.get_order(order_id)
    if not order:
        return 'order not found'
    # delivered, shipped or cancelled lo vuntey we cannot cancel an order

    if not order.can_modify:
        return f'cannot modify the order - current status is {order.status.value}'
    
    if ctx.deps.order_db_service.update_order_status(order_id,OrderStatus.CANCELLED):
        return f'Successfully cancelled order'
    
    return 'Failed to cancel the order'

def request_return(ctx:RunContext[Deps],order_id:str,reason:ReturnReason):
    """Process a return request for an order"""

    order=ctx.deps.order_db_service.get_order(order_id)
    if not order:
        return 'Order not found'
    if not order.can_return:
        if order.status!=OrderStatus.DELIVERED:
            return f'cannot return order current status is {order.status.value}'
        return 'cannor return order outside our 30 day window'
    
    return_id="RET-"+uuid4().hex[:12]

    return (
        f"return request approved for the order {order_id} \n"
        f"Reason:{reason.value}"
        f"A return label with ID {return_id } has been emailed to you,Please ship items within 14 days with all original tags attached"
    )

def escalate_to_human(ctx:RunContext[Deps],reason:EscalationReason,high_priority:bool=True):
    """Escalate the conversation to human set high_priority=True for urgent matters or when customer is dissatisfied"""
    response_time="1 hour" if high_priority else "24 Hours"
    return f" This matter has been escalated to our support team we wil contact you with in {response_time}"





    
    

class OrderResponse(BaseModel):
    thought:Annotated[str,...,'Your thougts to execute the task']
    response:Annotated[str,...,"The generated Response"]



def create_order_agent():
    model=build_model(model_name=settings.llm.name,api_key=settings.llm.api_key,base_url='',temperature=settings.llm.temperature,max_tokens=settings.llm.max_tokens)
    order_agent=Agent(model=model,system_prompt=settings.llm.prompt,tools=[get_order_details,update_shipping_address,cancel_order,request_return,escalate_to_human],result_type=OrderResponse,retries=3)
    return order_agent


def main():
    deps=Deps(order_db_service=OrderDBService())

    user_message='Requesting a return of item shipped as part of order 002, reason being wrong size'#'Cancel my order 001'
    order_agent=create_order_agent()
    result=order_agent.run_sync(user_message,deps=deps)
    print(result.output)
    print(deps.order_db_service.get_orders())



if __name__=='__main__':
    print(main())


    
