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


def cost(result):

    input_cost_per_million=0.15
    output_cost_per_million=0.60

    request_tokens=result.request_tokens
    response_tokens=result.response_tokens

    input_cost=(request_tokens/1000000)* input_cost_per_million
    output_cost=(response_tokens/1000000)* output_cost_per_million

    total_cost=input_cost+output_cost

    total_calls=result.requests


    print(f"Total input tokens were {request_tokens} and output tokens were {response_tokens}")
    print(f"total cost for this request was ${total_cost:.4f}")
    print(f"total calls to llm was {total_calls}")



client=AsyncOpenAI(api_key=os.getenv('GITHUB_TOKEN'),base_url='https://models.github.ai/inference')
provider=OpenAIProvider(openai_client=client)
model=OpenAIModel(model_name=os.getenv('GITHUB_MODEL'),provider=provider)
agent=Agent(model=model,system_prompt='You are a helpful assistant',retries=2)


try: 

    result=agent.run_sync("what is the capital of india")
    # print(f'result usage is : {result.usage},{result.usage().request_tokens}, {result.usage().response_tokens},{result.usage().requests}')
    cost(result.usage())

except ModelHTTPError as e:
    print(e)

except UnexpectedModelBehavior as e:
    print(e.__cause__)
    print(e)

else:
    print(result.output)





