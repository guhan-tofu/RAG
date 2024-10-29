# from langchain.document_loaders import DirectoryLoader
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# # from langchain.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from pinecone import Pinecone, ServerlessSpec
# import openai 
from dotenv import load_dotenv
import os
# import shutil


load_dotenv()

# Access your NVIDIA API key
#nvidia_api_key = os.environ.get('NVIDIA_API_KEY')

import getpass

if not os.getenv("NVIDIA_API_KEY"):
    # Note: the API key should start with "nvapi-"
    os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter your NVIDIA API key: ")

MODEL="nvidia/llama-3.1-nemotron-70b-instruct"

from langchain_openai.chat_models import ChatOpenAI

model = ChatNVIDIA(model = MODEL)
result = model.invoke("Tell me a small joke")
print(result.content)
