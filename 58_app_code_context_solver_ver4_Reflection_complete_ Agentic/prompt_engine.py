from problem import *
from code_retriever_agent import * 


def prompt_for_problem(problem: Problem) -> str:
    return f"""
    # Problem Description:
    {problem.description}
    # Input Specification:
    {problem.input_spec}
    # Output Specification:
    {problem.output_spec}
    # Sample Inputs:
    {problem.sample_inputs}
    # Sample Outputs:
    {problem.sample_outputs}
    # Note:
    {problem.notes}
    # Take input from:
    {problem.input_from}
    # Give output to:
    {problem.output_to}
    """


def get_retriever_agent_prompt(problem:Problem):
    return prompt_for_problem(problem)


def get_plan_generator_agent_prompt(similar_problem:ProblemInfo,problem:Problem,algo_technique:str):   
    return f"""
            # problem to be solved:{prompt_for_problem(problem)}
            # similar problem description : {similar_problem.description}
            # similar problem plan :{similar_problem.plan}
            #relevant algorithm technique to be used: {algo_technique}

            """


def get_plan_verifier_agent_prompt(problem:Problem,plan:str):
    return f"""
                #Problem to be solved: {prompt_for_problem(problem)}
                # plan for the problem : {plan}

            """

    


def get_code_generator_agent_prompt(problem:Problem,plan:str):

    return f"""#Problem to be solved: {prompt_for_problem(problem)} 
                #plan: {plan}"""


def get_code_improver_agent_prompt(problem:Problem,code:str,feedback:str,plan:str):

    return f"""
            # problem that your were solving: {prompt_for_problem(problem)}
            # plan: {plan}
            # buggy code : {code}
            # Test Report : {feedback}
            """

