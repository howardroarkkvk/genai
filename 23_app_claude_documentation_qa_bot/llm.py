
import os
from pydantic_ai.models import KnownModelName,Model
from typing import cast

def build_from_model_name(model_name:KnownModelName):
    print(model_name)
    if model_name.startswith("openai:"):
        from pydantic_ai.models.openai import OpenAIModel,OpenAIModelName
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIModel(cast(OpenAIModelName,model_name[7:]),provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY')))
    

    # elif model_name.startswith("anthropic:"):
        
    #     from pydantic_ai.models.anthropic import AnthropicModel,AnthropicModelName
    #     return AnthropicModel(cast(AnthropicModelName,model_name[10:]),api_key=os.getenv(''))
    
    elif model_name.startswith('google:gla'):
        from pydantic_ai.models.gemini import GeminiModel,GeminiModelName
        from pydantic_ai.providers.google_gla import GoogleGLAProvider

        return GeminiModel(cast(GeminiModelName,model_name[11:]),provider=GoogleGLAProvider(api_key=os.getenv('GEMINI_API_KEY')))
    
    elif model_name.startswith('groq:'):
        from pydantic_ai.models.groq import GroqModel,GroqModelName
        
        return GroqModel(model_name=cast(GroqModelName,model_name[5:]),api_key=os.getenv('GROQ_API_KEY'))
    
    elif model_name.startswith('mistral:'):
        from pydantic_ai.models.mistral import MistralModel,MistralModelName

        return MistralModel(cast(MistralModelName,model_name[8:]),api_key=os.getenv('MISTRAL_API_KEY'))
    
    elif model_name.startswith('ollama:'):
        print("is it reacching this point")
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider
        print(model_name)

        return OpenAIModel(model_name[7:],provider=OpenAIProvider(base_url='http://localhost:11434/v1'))
    
    else:
        raise ValueError(f"Unsupported model name :{model_name}")

    

    

