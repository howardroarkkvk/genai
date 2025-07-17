import gradio as gr
from summarizer import TextSummarizer

summarizer=TextSummarizer()

async def chatbot(text):
    # result=await summarizer.summarize_short_docs(text)
    result=await summarizer.summarize(text)
    return result


iface=gr.Interface(fn=chatbot,
                   inputs=[gr.Textbox(label='Enter text')],
                   title ='Text summarizer ChatBot',
                   description='provide the text and get summary',
                   outputs=[gr.Textbox(label='Model Response')]
                   )

iface.launch(share=False)



