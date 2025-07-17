from config_reader import * 
import logfire
import time
from agent import text_classifier_agent


class TextClassifier:
    def __init__(self):
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_openai()

    def predict(self,query):
        response=text_classifier_agent.run_sync(query)
        return response.data
    
if __name__=='__main__':
    textclassfier=TextClassifier()
    query = "I'm confused about a charge on my recent auto insurance bill that's higher than my usual premium payment."
    print(textclassfier.predict(query))
