from code_executor import * 
import logfire
from config_reader import * 
import time
from problem import * 
from prompt_engine import * 
from code_generator_agent import * 
from parser import * 
from code_improver_agent import * 
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from ingestor import * 

class CodeContestSolverDirect:
    def __init__(self):
        self.vector_db_dir=os.path.join(settings.file_paths.src_dir,settings.file_paths.vector_db_dir)
        self.embedding_model=HuggingFaceEmbeddings(model_name=settings.embedder.model,encode_kwargs={"normalize_embeddings": settings.embedder.normalize},model_kwargs={"token": settings.embedder.token},)
        self.reranker_model=HuggingFaceCrossEncoder(model_name=settings.reranker.model)

        if not os.path.exists(self.vector_db_dir):
            ingestor=KnowledgeIngestor()
            ingestor.ingest()
        else:
            self.vector_db=FAISS.load_local(folder_path=self.vector_db_dir,embeddings=self.embedding_model,allow_dangerous_deserialization=True)
        self.code_executor=CodeExecutor()
        logfire.configure(token=settings.logfire.token)
        time.sleep(2)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()


    async def generate(self,problem:Problem):

        deps=Deps(vector_db=self.vector_db,query=problem.description,reranker_model=self.reranker_model,retriever_top_k=settings.retriever.top_k,
             reranker_top_k=settings.reranker.top_k )
        
        code_generator_agent_prompt=get_code_generator_agent_prompt(problem=problem)
        code_generator_agent_result=await code_generator_agent.run(code_generator_agent_prompt,deps=deps)
        print(code_generator_agent_result)
        code=parse_response(code_generator_agent_result.output)
        logfire.info(f"seed code: \n{code}")
        is_solved=False
        cur_iter=1
        while cur_iter<=settings.max_iter:
            (is_solved,feedback)=self.code_executor.evaluate_with_feedback(problem.src_uid,problem.unit_tests,code)
            if is_solved:
                break
            feedback='Code is not correct, please give the correct code'
            logfire.info(f'Imporvement Iteration : {cur_iter}')
            logfire.info(f'is solved : {is_solved}')
            logfire.info(f"feedback : {feedback}")
            code_improver_agent_prompt=get_code_improver_agent_prompt(problem=problem,code=code,feedback=feedback)
            logfire.info(f'code improver agent prompt {code_generator_agent_prompt}')
            code_improver_agent_result=await code_improver_agent.run(code_improver_agent_prompt,deps=deps)
            code=parse_response(code_improver_agent_result.output)
            logfire.info(f'Modified code: {code}')
            cur_iter+=1
        return (is_solved,"",code)








