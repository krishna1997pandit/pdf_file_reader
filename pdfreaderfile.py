from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI


import os
os.environ["OPENAI_API_KEY"]= "pls enter openai_api_key"

pdfreader=PdfReader("C:\\Users\\Lenovo\\Desktop\\lap.pdf")

from typing_extensions import Concatenate
#read text from pdf
raw_text=""
for i ,page in enumerate(pdfreader.pages):
    content= page.extract_text()
    if content:
        raw_text +=content
        
        
# print(raw_text)
text_splitter = CharacterTextSplitter(
    separator ="\n",
    chunk_size=800,
    chunk_overlap =200,
    length_function =len,
)
texts =text_splitter.split_text(raw_text)
# print(texts,len(texts))

embeddings =OpenAIEmbeddings()

document_search = FAISS.from_texts(texts,embeddings)
print(document_search)

chain= load_qa_chain(OpenAI(), chain_type="stuff")

query = "tell about skills"
docs = document_search.similarity_search(query)
print(chain.invoke(input={'input_documents': docs, 'question': query}))
