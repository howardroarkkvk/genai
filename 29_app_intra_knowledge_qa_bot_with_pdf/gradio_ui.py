import gradio as gr
from rag import IntraKnowledgeQABot

intra_knowledge_bot=IntraKnowledgeQABot()
intra_knowledge_bot.ingest()

def chatbot(input:str,retriever_top_k):
    retriever_top_k=int(retriever_top_k)
    context=intra_knowledge_bot.retrievecontext(input,retriever_top_k)
    result=intra_knowledge_bot.execute(input,retriever_top_k)
    return context,result


print("Launcing Gradio")

iface=gr.Interface(fn=chatbot,
             inputs=[gr.Textbox(label='query'),gr.Slider(minimum=1,maximum=15,value=3,step=1,label="retreiver_top_k")],
             examples=[        
                 ["which navigation systems are created in 2022?", "2"],
                ["how much budget allocated for 2022?", "2"],
                ["major milestones achieved in 2022?", "2"],
                ["number of women employees in 2022?", "2"],
                ["number of women employees, inclusive of women in administration, in 2022?","2",],
                ["milestones achieved in 2022?", "2"]],
            outputs=[gr.Textbox(label="context"),gr.Textbox(label='response')],
            title="Intra Knowledge QA Bot",
            description="Ask questions on intra knowledge base and get the answers.",
            )

iface.launch(share=False)



