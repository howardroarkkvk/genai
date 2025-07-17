

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# using hugging face cross encode model for reranking -> first we ahve to create the model
# then we have to set the object crossencodereranker using the hugging fact model
# using Contextual compression retriever ( where we pass the base retriever and cross reranker)
# here base retriever is created on the vector db
# using contextual compression retrtiever, we ahve to  invoke...

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever


class Retriever:
    def __init__(self,main_dir,vector_db_dir):
        self.embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
        self.rerank_model=HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-base')#os.getenv('HF_RERANKING_MODEL')
        # print(os.getenv('HF_RERANKING_MODEL'))
        self.main_dir=main_dir
        self.vector_db=FAISS.load_local(folder_path=vector_db_dir,embeddings=self.embedding_model,allow_dangerous_deserialization=True)

    def retriever(self,query:str,top_k:int):
        # print("with in retriever")
        base_retriever=self.vector_db.as_retriever(search_kwargs={'k':top_k})
        outputs=[]
        results=base_retriever.invoke(query)
        for result in results:
            outputs.append(result.metadata['source'])
        return outputs
    
    def retrive_with_rerank(self,query:str,top_k:int):
        print("with in rerank model")
        retriever=self.vector_db.as_retriever(search_kwargs={'k':10})
        compressor=CrossEncoderReranker(model=self.rerank_model,top_n=2)
        compression_retriever=ContextualCompressionRetriever(base_compressor=compressor,base_retriever=retriever)
        # print("Before results step")
        results=compression_retriever.invoke(query)
        outputs=[]
        for output in results:
            outputs.append(output.metadata['source'])
        return outputs
    

if __name__=='__main__':
    main_dir=r'D:\DataFiles\oops'
    vector_db_dir=r'D:\DataFiles\oops\faiss_oops'
    retrieve=Retriever(main_dir=main_dir,vector_db_dir=vector_db_dir)

    queries = [
        "How can you create multiple test cases for an evaluation in the Anthropic Evaluation tool?",
        "What embeddings provider does Anthropic recommend for customized domain-specific models, and what capabilities does this provider offer?",
    ]
    for query in queries:
        output=retrieve.retriever(query,2)
        print(output)

    for query in queries:
        output=retrieve.retrive_with_rerank(query,2)
        print(output)   

















