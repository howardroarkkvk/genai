import gradio as gr

def greet(name,adress):
    Name='Vamshi' 
    Address= 'Bachupally' if name=='vamshi' else 'kukatpally'
    return Name,Address


demo=gr.Interface(fn=greet,inputs=gr.Textbox(label='Name'),title='First Web Page',description="Gradio can be used for LLM outputs",
                  examples=['vamshi','Ashwini'],outputs=[gr.Textbox(label='Name'),gr.Textbox(label='Address')],theme=gr.themes.soft)

demo.launch()