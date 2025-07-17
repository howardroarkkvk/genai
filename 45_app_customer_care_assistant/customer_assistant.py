from config_reader import *
import logfire
import time
from triage_agent import *
class CustomerAssistant:
    def __init__(self):
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_openai()

    async def process(self,query:str):
        print("It's in process ")
        result=await triage_agent.run(query)
        return result.output
    

async def main(query:str):
    ca=CustomerAssistant()
    result=await ca.process(query)
    print(result)


if __name__=='__main__':
    query="what is the  balance of the customer with id=123"
    #"what is the loan status of customer with id=123"
    # query = "what is the account balance"
    asyncio.run(main(query))


