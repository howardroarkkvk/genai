

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker


load_dotenv(override=True)

embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
vector_db_path=r'D:\faiss2'
vector_db=FAISS.load_local(folder_path=vector_db_path,embeddings=embedding_model,allow_dangerous_deserialization=True)

queries = [
    "How long does international shipping take?",
    "Show the Leather jackets in the store",
    "Contact details of store",
    "What is the return policy?",
]

print(os.getenv('HF_RERANKING_MODEL'))
rerank_model=HuggingFaceCrossEncoder(model_name=os.getenv('HF_RERANKING_MODEL'))
for query in queries:
 
    retriever=vector_db.as_retriever(search_kwargs={'k':10})
    compressor=CrossEncoderReranker(model=rerank_model,top_n=2)
    compression_retriever=ContextualCompressionRetriever(base_compressor=compressor,base_retriever=retriever)
    hits=compression_retriever.invoke(query)
    print('\n Query:',query)
    print('Top 2 similar chunks from the knowledge base')
    for hit in hits:
        print(hit.page_content,
              f'Source:{hit.metadata['source']}')
        print()
