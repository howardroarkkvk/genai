import gradio as gr

import json
from problem import * 
from code_contest_solver import * 


solver=CodeContestSolverDirect()

async def respond(description, unit_tests):
    print(type(description))
    print(type(unit_tests))
    desc=json.loads(description)
    # unit_tests=json.loads(unit_tests)
    # print(unit_tests)
    problem=Problem(src_uid=desc['src_uid'],
                    description=desc['description'],
                    input_spec=desc['input_spec'],
                    output_spec=desc['output_spec'],
                    sample_inputs=desc['sample_inputs'],
                    sample_outputs=desc['sample_outputs'],
                    notes=desc['notes'],
                    unit_tests=json.loads(unit_tests)
                    )
    print(problem.unit_tests)
    (is_solved,plan,code)=await solver.generate(problem)



    return plan,code,str(is_solved)


iface=gr.Interface(fn=respond,
             inputs=[gr.Textbox(label='Description',lines=10),gr.Textbox(label='Unit tests',lines=5)],
            title='Code Contest Solver',    
            description=" Provide your contest problem with unit test cases and get your code",
            outputs=[gr.Textbox(label='Plan',lines=5),
                     gr.Textbox(label='Code',lines=5),
                     gr.Textbox(label='Is_Solved',lines=1)]

            
            )

iface.launch(share=False)



