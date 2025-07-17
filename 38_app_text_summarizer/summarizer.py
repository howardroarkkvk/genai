from config_reader import *
from agent import text_summarizer_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio
import logfire
import os
import time
from sklearn.cluster import KMeans
import numpy as np

class TextSummarizer:
    def __init__(self):
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=settings.chunker.chunk_size,chunk_overlap=settings.chunker.chunk_overlap,length_function=len)
        self.embedding_model=HuggingFaceEmbeddings(model_name=settings.embedder.model,encode_kwargs={'normalize_embeddings':settings.embedder.normalize},model_kwargs={'token':settings.embedder.token})
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_openai()


    async def summarize_short_docs(self,text:str):
        response=await text_summarizer_agent.run(text)
        return response.data
    

    async def summarize_medium_docs(self,text:str=None,chunks:list[str]=None):
        if chunks==None:
            chunks=self.text_splitter.split_text(text)
        co_routine_tasks_of_texts=[text_summarizer_agent.run(text) for text in chunks]
        results=await asyncio.gather(*co_routine_tasks_of_texts)
        chunks_result_data_list=[result.data for result in results]
        total_summarized_text=''.join(chunks_result_data_list)
        response=await text_summarizer_agent.run(total_summarized_text)
        return response.data
    
    async def summarize_high_docs(self,text:str=None):
        chunks=self.text_splitter.split_text(text)
        batch_size=2
        vectors=[]
        for i in range(0,len(chunks),batch_size=2):
            vectors.extend(self.embedding_model.embed_documents(chunks[i:i+batch_size]))

        n_clusters=5
        model=KMeans(n_init=10,n_clusters=5,random_state=42).fit(vectors)

        closest_indices=[]
        for i in n_clusters:
            distances=np.linalg.norm(vectors - model.cluster_centers_[i])
            min_indexes=np.argmin(distances)
            closest_indices.append(min_indexes)
        selected_indices=sorted(closest_indices)
        selected_chunks=[chunks[idx] for idx in selected_indices]
        return await self.summarize_medium_docs(selected_chunks)
    
    async def summarize(self,text:str=None):
        if len(text)<=settings.docs.short_doc_threshold:
            return await self.summarize_short_docs(text)
        if len(text)<=settings.docs.medium_doc_threshold:
            return await self.summarize_medium_docs(text)
        if len(text)<=settings.docs.high_doc_threshold:
            return await self.summarize_high_docs(text)
        else:
            return 'not supported'
        



        

