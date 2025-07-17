import gradio as gr

from classifier import TextClassifier

classifier=TextClassifier()

def chatbot(input_text):
    response=classifier.predict(input_text)
    return response


iface=gr.Interface(fn=chatbot,
             inputs=[gr.Textbox(label='Query')],
             examples=[
        "I'm confused about a charge on my recent auto insurance bill that's higher than my usual premium payment. Can you clarify what this extra fee is for and why it was added?",
        "I came out to my car after shopping and found that another vehicle had hit it, but the driver was nowhere to be found. What should my next steps be in terms of filing an insurance claim for the damage?",
    ],
             title="Insurance Support Ticket Classifier Bot",
             description="Provide the ticket details and get the class of the ticket.",
             outputs=[gr.Textbox(label='Reponse')]
)


iface.launch(share=False)