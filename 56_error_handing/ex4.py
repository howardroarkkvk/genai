from openai import AsyncOpenAI
from pydantic_ai import Agent,ModelRetry,UnexpectedModelBehavior,capture_run_messages
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.groq import GroqProvider
import os
from pydantic_ai.exceptions import ModelHTTPError
from dotenv import load_dotenv
import logfire
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.fallback import FallbackModel

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()
logfire.instrument_pydantic_ai()



client=AsyncOpenAI(api_key=os.getenv('GITHUB_TOKEN'),base_url='https://models.github.ai/inference')
github_provider=OpenAIProvider(openai_client=client)
github_model=OpenAIModel(model_name=os.getenv('GITHUB_MODEL'),provider=github_provider)
groq_provider=GroqProvider(api_key=os.getenv('GROQ_API_KEY'))
groq_model=GroqModel(model_name=os.getenv('GROQ_CHAT_MODEL'),provider=groq_provider)
fallback_model=FallbackModel(groq_model,github_model)


agent=Agent(model=fallback_model,system_prompt='You are a helpful assistant',retries=2)


@agent.tool_plain
def calc_volume(size:int):
    """"Calculates the volume of the box
    
    Args:
    size: size of the box

    """

    if size ==42:
        return 42**3
    else:
        raise ModelHTTPError(status_code=200,model_name=os.getenv('GROQ_MODEL_NAME'))
    


try: 

    result=agent.run_sync("Please get the volume of the box with size 6")

except ModelHTTPError as e:
    print(e)

except UnexpectedModelBehavior as e:
    print(e.__cause__)
    print(e)

else:
    print(result.output)





