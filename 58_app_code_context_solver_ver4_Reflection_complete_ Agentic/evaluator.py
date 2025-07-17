import asyncio
import json
import os
from config_reader import * 
from problem import * 
from code_contest_solver import *

async def main():
    eval_file=os.path.join(settings.file_paths.src_dir,settings.file_paths.eval_file)
    list_of_jsonl_lines=[]
    solver=CodeContestSolverDirect()
    with open(eval_file,'r',encoding='utf-8') as f:
        for line in f:
            data=json.loads(line)
            list_of_jsonl_lines.append(data)
    
    results=[]
    for p in list_of_jsonl_lines:
        problem=Problem(
                src_uid=p['src_uid'],
                description=p['prob_desc_description'],
                input_spec=p['prob_desc_input_spec'],
                output_spec=p['prob_desc_output_spec'],
                sample_inputs=p['prob_desc_sample_inputs'],
                sample_outputs=p['prob_desc_sample_outputs'],
                notes=p['prob_desc_notes'],
                unit_tests=json.loads(p['hidden_unit_tests'])
                )
        res=await solver.generate(problem)
        results.append(res)
    results_dir=os.path.join(settings.file_paths.src_dir,settings.file_paths.results_file.split('/')[0])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file=os.path.join(settings.file_paths.src_dir,settings.file_paths.results_file)
    with open(results_file,'w',encoding='utf-8') as f:
        for line in results:
            f.write(json.dumps(line+'\n'))


if __name__=='__main__':
    print(asyncio.run(main()))



