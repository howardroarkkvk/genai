
from classifier import TextClassifierRAG
from config_reader import settings
import os
import json 
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ClassificationEvaluator:
    def __init__(self):
        self.src_dir=settings.file_paths.src_dir
        self.eval_file =settings.file_paths.eval_file
        self.eval_file_with_response = settings.file_paths.eval_file_with_response
        self.results_dir = os.path.join(self.src_dir,settings.file_paths.results_dir)
        self.results_report_file=settings.file_paths.results_report_file
        self.results_matrix_file=settings.file_paths.results_matrix_file

    def generate_response(self):
        classifier=TextClassifierRAG()
        with open(os.path.join(self.src_dir,self.eval_file),'r') as f:
            eval_data=json.load(f)

        docs=[]
        for item in eval_data:
            tmp={'id':item['id'],
                 'question':item['question'],
                 'model_response':classifier.predict(item['question']),
                 'reference':item['answer']}
            docs.append(tmp)
        
        with open(os.path.join(self.src_dir,self.eval_file_with_response),'w') as f:
            json.dump(docs,f,indent=2)
        print(">>> end of Generating response for evalation metrics")

    def generate_metrics(self):
        print('Begin : Evaluating metrics')
        with open(os.path.join(self.src_dir,self.eval_file_with_response),'r') as f:
            eval_data=json.load(f)
        
        responses=[]
        references=[]
        for item in eval_data:
            responses.append(item['model_response'])
            references.append(item['reference'])

        report=classification_report(references,responses,output_dict=True)
        df=pd.DataFrame(report)#.transpose()
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        df.to_csv(os.path.join(self.results_dir,self.results_report_file))

        labels=sorted(list(set(references)))
        cm=confusion_matrix(references,responses,labels=labels)
        print(cm)  
        self.save_confusion_matrix(cm, labels) 

        print("End of generating metrics")


    def save_confusion_matrix(self,cm=np.array([[1,2],[3,4]]),labels=['a','b','c']):
        fig,ax=plt.subplots(figsize=(10,10))
        
        # creating heatmap
        im=ax.imshow(cm,cmap='Blues')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        
        # Set tick labels and positions
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        # Add labels to each cell
        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        # Set labels and title
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, self.results_matrix_file))



if __name__=='__main__':
    classifier_evaluator=ClassificationEvaluator()
    # classifier_evaluator.generate_response()
    classifier_evaluator.generate_metrics()
    # classifier_evaluator.save_confusion_matrix()



