
import json
import os

from config_reader import settings
from rag import TextToSqlRAG
from ragas import SingleTurnSample,EvaluationDataset
from langchain_groq import ChatGroq
from ragas.metrics import LLMSQLEquivalence
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate,RunConfig
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from utils import execute_query

class TextToSqlEvaluator:
    def __init__(self):
        self.src_dir=settings.file_paths.src_dir
        self.eval_file=settings.file_paths.eval_file
        self.eval_file_with_response=settings.file_paths.eval_file_with_response
        self.results_dir=settings.file_paths.results_dir
        self.db_dir=os.path.join(self.src_dir,settings.file_paths.db_file)
        llm=ChatGroq(model=settings.judge_llm.name, api_key=settings.judge_llm.api_key)
        self.evaluator_llm=LangchainLLMWrapper(llm)
        self.embedding_model=HuggingFaceEmbeddings(model_name=settings.embedder.model,encode_kwargs={'normalize_embeddings':settings.embedder.normalize},model_kwargs={'token':settings.embedder.token})






    def generate_llm_responses(self):
            
            print(">> Begining of Generate LLM reponese")

            with open(os.path.join(self.src_dir,self.eval_file),'r') as f:
                jsondata_extract_as_list=json.load(f)

            rag = TextToSqlRAG()
            rag.chunk_schema()
            rag.ingest()
            responses_list=[]
            for dict_values in jsondata_extract_as_list:
                #  print(dict_values)
                 response={'id':dict_values['id'],
                           'input_question':dict_values['question'],
                           'llm_answer':rag.execute(dict_values['question']),
                           'retrieved_context':rag.retrievecontext(dict_values['question']),
                            'eval_answer':dict_values['answer']
                           }
                 responses_list.append(response)
            # print(responses_list)

            with open(os.path.join(self.src_dir,self.eval_file_with_response),'w') as f:
                 json.dump(responses_list,f,indent=2)

            print("End of LLM reponses generation")

    def evaluate_llm_metrics(self):

        print(">> Start of LLM evalation")
                
        with open(os.path.join(self.src_dir,self.eval_file_with_response),'r') as f:
              eval_file_responses=json.load(f)
        # step 1 creation of samples list using Single Turn Sample function
        samples=[]
        for responses in eval_file_responses:
            sample=SingleTurnSample(user_input=responses['input_question'],
                                    reference_contexts=responses['retrieved_context'],
                                    response=responses['llm_answer'],
                                    reference=responses['eval_answer'])
            samples.append(sample)

        # step 2 creation of evalaution dataset using the created samples
        evaluation_dataset=EvaluationDataset(samples)

        # step 3: creation of metrics ....
        sql_equivalence_metric=LLMSQLEquivalence(llm=self.evaluator_llm)

        # step 4: evaluating
        result=evaluate(dataset=evaluation_dataset,metrics=[sql_equivalence_metric],embeddings=self.embedding_model,llm=self.evaluator_llm,run_config=RunConfig(max_workers=4,max_wait=60),show_progress=True)


        results_dir_path=os.path.join(self.src_dir,self.results_dir)
        if not os.path.exists(results_dir_path):
             os.makedirs(results_dir_path)

        print(result)

        df=result.to_pandas()
        df.to_csv(os.path.join(results_dir_path,'results_llm_metrics.csv'), index=False)

        print(">> end of llm evalation")


    def evaluate_non_llm_response(self):


        print(">> Start of Evaluator Non LLM reponses :")
         
        
        with open(os.path.join(self.src_dir,self.eval_file_with_response),'r') as f:
            eval_file_with_llm_respnse=json.load(f)
        
        evaluator_query_list=[]
        llm_generated_query_list=[]

        for item in eval_file_with_llm_respnse:
             llm_generated_query_list.append(execute_query(self.db_dir,item['llm_answer']))
             evaluator_query_list.append(execute_query(self.db_dir,item['eval_answer']))

        results_dir_path=os.path.join(self.src_dir,self.results_dir)
        if not os.path.exists(results_dir_path):
             os.mkdirs(results_dir_path)

        df=pd.json_normalize(eval_file_with_llm_respnse)
        df['llm_query_answer']=llm_generated_query_list
        df['eval_file_query_answer']=evaluator_query_list

        df.to_csv(os.path.join(results_dir_path,'non_llm_results.csv'),index=False)



        print(">> End of Evaluator Non LLM reponses :")


    
        


        
    
        
        
        
              

        
         



if __name__=='__main__':
     ttse=TextToSqlEvaluator()
     ttse.evaluate_non_llm_response()
    #  ttse.generate_llm_responses()
    #  ttse.evaluate_llm_metrics()

                 
    


