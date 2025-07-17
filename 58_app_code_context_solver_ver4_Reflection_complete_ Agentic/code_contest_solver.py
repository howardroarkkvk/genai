from code_executor import * 
import logfire
from config_reader import * 
import time
from problem import * 
from prompt_engine import * 
from code_generator_agent import * 
from parser import * 
from code_improver_agent import * 
from code_retriever_agent import * 
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from ingestor import * 
from typing import List
from code_plan_generator_agent import * 
from code_plan_verifier_agent import * 


class CodeContestSolverDirect:

    def __init__(self):
        self.code_executor = CodeExecutor()
        logfire.configure(token=settings.logfire.token)
        time.sleep(2)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    async def retreive_simialar_problems(self,problem:Problem):
        retriever_agent_prompt=get_retriever_agent_prompt(problem=problem)
        result=await retriever_agent.run(retriever_agent_prompt)
        for i,p in enumerate(result.output.similar_problems):
            logfire.info(f"similar problem ={i+1}")
            logfire.info(f"Description : {p.description}")
            logfire.info(f"code: {p.code}")
            logfire.info(f"planning : {p.plan}")

        logfire.info(f"algorithm technique to be used: {result.output.algorithm_technique}")

        return (result.output.similar_problems,result.output.algorithm_technique)
    
    async def generate_plans_with_verified_scores(self,problem:Problem,similar_problems:List[ProblemInfo],alogrith_technique:str):
        plans=[]
        for _,similar_prob in enumerate(similar_problems):
            plan_generator_agent_prompt=get_plan_generator_agent_prompt(similar_problem=similar_prob,problem=problem,algo_technique=alogrith_technique)
            plan_generator_agent_result=await plan_generator_agent.run(plan_generator_agent_prompt)
            plan_verifier_agent_prompt=get_plan_verifier_agent_prompt(problem=problem ,plan =plan_generator_agent_result.output)
            plan_verifier_agent_result=await plan_verifier_agent.run(plan_verifier_agent_prompt)
            plans.append((plan_generator_agent_result.output,plan_verifier_agent_result.output.confidence_score,similar_prob))
        
        plans.sort(key=lambda x:x[1],reverse=True)

        for i,plan in enumerate(plans):
            logfire.info(f'plan {i+1} , confidence_score: {plan[1]}')
            logfire.info(f'plan[0]')
        return plans
    

    async def generate_code_with_feedback_loop(self,problem:Problem,plan:str):

        code_generator_agent_prompt=get_code_generator_agent_prompt(problem=problem,plan=plan)
        code_generator_agent_result=await code_generator_agent.run(code_generator_agent_prompt)
        code=parse_response(code_generator_agent_result.output)
        logfire.info(f"seed code is : {code}")
        is_solved=False
        cur_iter=1

        while cur_iter<settings.max_iter:
            is_solved,feedback=self.code_executor.evaluate_with_feedback(problem.src_uid,problem.unit_tests,code)
            if is_solved:
                break
            logfire.info(f"improvement iter: {cur_iter}")
            logfire.info(f"is solved: {is_solved}")
            logfire.info(f"feedback received: {feedback}")

            code_improver_agent_promopt=get_code_improver_agent_prompt(problem=problem,code=code,feedback=feedback,plan=plan)
            code_improver_agent_result=await  code_improver_agent.run(code_improver_agent_promopt)
            code=parse_response(code_improver_agent_result.output.modified_code)
            plan=code_improver_agent_result.output.modified_plan

            logfire.info(f"current plan is : {code_improver_agent_result.output.current_plan}")
            logfire.info(f"modified code is : {code}")
            logfire.info(f"modified plan is : {plan}")

            cur_iter+=1

        return (is_solved,plan,code)
    


    async def generate(self,problem:Problem):
        (similar_problems,algo_technique)=await self.retreive_simialar_problems(problem)
        plans=await self.generate_plans_with_verified_scores(problem=problem,similar_problems=similar_problems,alogrith_technique=algo_technique)
        final_plan=''
        final_code=''
        for i,plan in enumerate(plans):
            (is_solved, final_plan, final_code)=await self.generate_code_with_feedback_loop(problem=problem,plan=plan[0])
            if is_solved:
                break

        return (is_solved,final_plan,final_code)





    













        















