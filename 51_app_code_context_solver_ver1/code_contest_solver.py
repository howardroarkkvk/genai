from code_executor import * 
import logfire
from config_reader import * 
import time
from problem import * 
from prompt_engine import * 
from code_generator_agent import * 
from parser import * 

class CodeContestSolverDirect:
    def __init__(self):
        self.code_executor=CodeExecutor()
        logfire.configure(token=settings.logfire.token)
        time.sleep(2)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()


    async def generate(self,problem:Problem):
        is_solved=False
        cur_iter=1
        while cur_iter<=settings.max_iter and not is_solved:
            logfire.info(f'Iteration : {cur_iter}')
            logfire.info('Generating code')
            code_generator_agent_prompt=get_code_generator_agent_prompt(problem=problem)
            code_generator_agent_result=await code_generator_agent.run(code_generator_agent_prompt)
            code=parse_response(code_generator_agent_result.output)

            logfire.info(f"code:{code}")

            logfire.info(f'Executing code')
            is_solved=self.code_executor.evaluate(problem.src_uid,problem.unit_tests,code)
            logfire.info(f"exec result: {is_solved}")
            cur_iter+=1
        return (is_solved,"",code)








