from order_processing_agent import *
import logfire
from config_reader import *
from pydantic_ai.messages import ModelMessage
from order_db_service import *

# we can use order_processing agent or we can use orger_assistant , in order rocessing agent main() will do what process in this program does...
class OrderAssistant:
    def __init__(self):
        logfire.configure(token=settings.logfire.token)
        logfire.instrument_openai()
        self.order_agent=create_order_agent()
    
    def process(self,user_query:str,history:list[ModelMessage]):
        deps=Deps(order_db_service=OrderDBService())
        result=self.order_agent.run_sync(user_query,deps=deps,message_history=history)
        logfire.info(f'thoughts:{result.output.thoughts}')
        return result.output
