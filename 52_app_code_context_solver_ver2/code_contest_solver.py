from code_executor import * 
import logfire
from config_reader import * 
import time
from problem import * 
from prompt_engine import * 
from code_generator_agent import * 
from parser import * 
from code_improver_agent import * 

class CodeContestSolverDirect:
    def __init__(self):
        self.code_executor=CodeExecutor()
        logfire.configure(token=settings.logfire.token)
        time.sleep(2)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()


    async def generate(self,problem:Problem):
        
        code_generator_agent_prompt=get_code_generator_agent_prompt(problem=problem)
        code_generator_agent_result=await code_generator_agent.run(code_generator_agent_prompt)
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
            code_improver_agent_result=await code_improver_agent.run(code_improver_agent_prompt)
            code=parse_response(code_improver_agent_result.output)
            logfire.info(f'Modified code: {code}')
            cur_iter+=1
        return (is_solved,"",code)








