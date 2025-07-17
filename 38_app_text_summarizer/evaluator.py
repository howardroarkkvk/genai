from pydantic_evals import Case,Dataset
from pydantic_evals.evaluators import Evaluator,EvaluatorContext
from bert_score import score
from rouge import Rouge
from config_reader import *
import os
from summarizer import TextSummarizer
import json
import asyncio
from io import StringIO
from rich.console import Console

class BertEvaluator(Evaluator):
    def evaluate(self,ctx:EvaluatorContext[str,str]):
        p,r,f=score([ctx.output],[ctx.expected_output],lang='en')
        return f.item()


class RougeEvaluator(Evaluator):
    def evaluate(self,ctx:EvaluatorContext[str,str]):
        scores=Rouge().get_scores(ctx.output,ctx.expected_output)
        rouge_1=scores[0]['rouge-1']
        return rouge_1


class SummarizationEvaluator:
    def __init__(self):
        self.src_dir=settings.file_paths.src_dir
        self.eval_file=settings.file_paths.eval_file
        self.results_dir=settings.file_paths.results_dir
        self.results_dir_path=os.path.join(self.src_dir,self.results_dir)
        self.results_non_llm_metrics_file=settings.file_paths.results_non_llm_metrics_file
        if not os.path.exists(self.results_dir_path):
            os.makedirs(self.results_dir_path)
        self.summarizer=TextSummarizer()


    async def evaluate(self):
        print(">>> Begin Evaluating LLM metrics")
        with open(os.path.join(self.src_dir,self.eval_file),'r') as f:
            eval_data=json.load(f)
        
        cases=[]
        for item in eval_data:
            case=Case(name=str(item['id']),
                 inputs=item['text'],
                 expected_output=item['summary'])
            cases.append(case)

        dataset=Dataset(cases=cases,evaluators=[BertEvaluator(),RougeEvaluator()])
        report=await dataset.evaluate(self.summarizer.summarize)
        table=report.console_table(include_input=True,
                             include_output=True,
                             include_expected_output=True,
                             include_averages=True
                             )
        
        
        with open(os.path.join(self.src_dir,self.results_non_llm_metrics_file),'w',encoding='utf-8') as file:
            io_file=StringIO() # in memory buffer...
            Console(file=io_file).print(table)
            # print(io_file.getvalue()) #using the getvalue() method we retrieve all the details from the in memory bufferr...
            # how to load from in memory bufferr to file....file.write(io_file.get_value())
            file.write(io_file.getvalue())

        print(">>>End of Evaluating LLM Metrics")



        






if __name__=='__main__':
    summarizer=SummarizationEvaluator()
    asyncio.run(summarizer.evaluate())

