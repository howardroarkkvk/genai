from openai import AsyncOpenAI
from pydantic_ai import Agent,ModelRetry,UnexpectedModelBehavior,capture_run_messages
from pydantic_ai.providers.openai import OpenAIProvider
import os
from pydantic_ai.exceptions import ModelHTTPError
from dotenv import load_dotenv
import logfire
from pydantic_ai.models.openai import OpenAIModel


load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()
logfire.instrument_pydantic_ai()



client=AsyncOpenAI(api_key=os.getenv('GITHUB_TOKEN'),base_url='https://models.github.ai/inference')
provider=OpenAIProvider(openai_client=client)
model=OpenAIModel(model_name=os.getenv('GITHUB_MODEL'),provider=provider)
agent=Agent(model=model,system_prompt='You are a helpful assistant',retries=2)


@agent.tool_plain
def calc_volume(size:int):
    """"Calculates the volume of the box
    
    Args:
    size: size of the box

    """

    if size==42:
        return size**3
    else:
        print(f'Invalid size: {size}')
        raise ModelRetry('Please try again with size 42')
    


try: 

    result=agent.run_sync("Please get the volume of the box with size 6")

except ModelHTTPError as e:
    print(e)

except UnexpectedModelBehavior as e:
    print(e.__cause__)
    print(e)

else:
    print(result.output)





