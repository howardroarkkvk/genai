from order_processing_agent import *
import logfire
from config_reader import *
from pydantic_ai.messages import ModelMessage
from order_db_service import *
from input_guardrail_agent import *

class OrderAssistant:
    def __init__(self):
        logfire.configure(token=settings.logfire.token)
        logfire.instrument_openai()
        self.order_agent=create_order_agent()

    def get_query_category(self,user_query:str):
        response=input_guard_rail_agent.run_sync(user_query)
        print(response)
        return response.output.ticket_class.value

    
    def process(self,user_query:str,history:list[ModelMessage]):
        deps=Deps(order_db_service=OrderDBService())        
        query_category=self.get_query_category(user_query)
        if query_category ==RequestClassEnum.OT:
            return 'Invalid Request'

        result=self.order_agent.run_sync(user_query,deps=deps,message_history=history)
        print('result in process',result.output.response)
        logfire.info(f'thoughts:{result.output.thought}')
        return result.output.response
