from multiprocessing import Process,Queue
import multiprocessing
import subprocess
from enum import Enum
import time
import traceback
import sys

class ExecOutcome(Enum):
    PASSED='PASSED' # code executes and output matches the expected output
    WRONG_ANSWER='WRONG_ANSWER' # code executes and output doesnt match with the expected output
    TIME_LIMIT_EXCEEDED='TIME_LIMIT_EXCEEDED' # code executes and didnt exit in time, output is ignored in this case
    RUNTIME_ERROR='RUNTIME_ERROR' # code failed to execute (crashed)
    COMPILATION_ERROR='COMPILATION_ERROR' # code failed to compile
    MEMORY_LIMIT_EXCEEDED='MEMORY_LIMIT_EXCEEDED' # code exceeded memory limit during execution


class CodeExecutor:

    def __init__(self):
        multiprocessing.set_start_method("spawn",force=True)


    def exec_program(self,q,program,input_data,expected_output,timeout):
        try:
            
            start_time=time.time()
            process=subprocess.Popen([sys.executable,"-c",program],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
            stdout,stderr=process.communicate(input=input_data,timeout=timeout)
            if time.time()-start_time>timeout:
                raise TimeoutError("Execution Timed Out") # time out error
            if process.returncode!=0:
                q.put(f'failed:{stderr}')  # run time error or compilation error
            else:
                if stdout.strip()==expected_output.strip():
                    q.put('passed') # successfully run 
                else:
                    q.put(f"wrong answer ,expected_output: '{expected_output}' got '{stdout}'") # wrong answer
        except subprocess.TimeoutExpired:
            process.kill()
            q.put("timed out") # timeout error

        except Exception: # any other exception while running the try block.....
            q.put(f'failed: {traceback.format_exc()}')


    def check_correctness(self,program:str,input_data:str,expected_output:str,timeout:float):
        q=Queue()
        process=Process(target=self.exec_program,args=(q,program,input_data,expected_output,timeout))
        process.start()
        process.join(timeout=timeout+15)
        if process.is_alive():
            process.terminate()
            process.join()
            result='Timed out'
        else:
            try:
                result=q.get_nowait()
            except Exception as e:
                result=f"the exception occured is {e}" 
        return result
    
    def evaluate(self,src_uid:str,unittests:list[dict],code:str):
        test_results=[]
        print('unittest is:',type(unittests),unittests)
        for test_case in unittests:
        
            print('test case is :',test_case)
            input_data=test_case['input']
            print(input_data)
            expected_output=test_case['output'][0]
            print(expected_output)
            test_result=self.check_correctness(program=code,input_data=input_data,expected_output=expected_output,timeout=2)
            test_results.append(test_result)
        for result in test_results:
            print(result)

        
        return (True if result else False)
    

    



        









