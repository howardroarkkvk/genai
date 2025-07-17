from config_reader import *
import os
from langchain_huggingface import HuggingFaceEmbeddings
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

class TextClassifier:
    def __init__(self):
        self.src_dir=settings.file_paths.src_dir
        self.train_file=os.path.join(self.src_dir,settings.file_paths.train_file)
        self.embedding_dir=os.path.join(self.src_dir,settings.file_paths.embeddings_dir)
        self.model_dir=os.path.join(self.src_dir,settings.file_paths.model_dir)
        self.embedding_model=HuggingFaceEmbeddings(model_name=settings.embedder.model,encode_kwargs={"normalize_embeddings": settings.embedder.normalize},model_kwargs={"token": settings.embedder.token})

        if not os.path.exists(self.embedding_dir):
            self.generate_embeddings()
            
        if not os.path.exists(self.model_dir):
            self.create_model()

    def generate_embeddings(self):
        print("Creation of Embeddings file on the train data json file")
        with open(self.train_file,'r') as f:
            eval_data=json.load(f)
        items=[]
        for item in eval_data:
            query_embedding=self.embedding_model.embed_query(item['question'])
            tmp={'question':item['question'],
                 'answer':item['answer'],
                 'query_embedding':query_embedding}
            items.append(tmp)
        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
        with open(os.path.join(self.embedding_dir,'train_embeddings.json'),'w') as f:
            json.dump(items,f,indent=2)

        print(">> End of creatoni of embedding files")

    def create_model(self):
        print(">> Creating Model ")

        with open(os.path.join(self.embedding_dir,'train_embeddings.json'),'r') as f:
            train_embedding_data=json.load(f)

        df=pd.json_normalize(train_embedding_data)
        clf=LogisticRegression(random_state=42)
        X=pd.DataFrame(df['query_embedding'].to_list())
        y=df['answer']
        clf.fit(X,y)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        joblib.dump(clf,os.path.join(self.model_dir,'embedding_model.pkl'))

        print('>> end of model training')


    def predict(self,query):
        input_query_embedding=np.array(self.embedding_model.embed_query(query))
        input_query_embedding=input_query_embedding.reshape(1,-1)
        model=joblib.load(os.path.join(self.model_dir,'embedding_model.pkl'))
        result=model.predict(input_query_embedding)
        return result[0]
    

if __name__=='__main__':
    classifier=TextClassifier()
    query = "I'm confused about a charge on my recent auto insurance bill that's higher than my usual premium payment."
    print(classifier.predict(query))

                
                






        

