# RAG Application: Retrieval-Augmented Generation with NVIDIA LLaMA 3.1, Pinecone, and MPNet

This project is a Retrieval-Augmented Generation (RAG) pipeline that leverages NVIDIA's **LLaMA 3.1** language model for natural language interaction, **Pinecone** for vector storage and retrieval, and the **all-mpnet-base-v2** model to create text embeddings. The application allows users to upload documents, process them into embeddings, store them in a vector database, and perform contextualized Q&A based on relevant retrieved documents.

## Features

- **Upload Document**: Supports .txt file uploads for document embedding and storage in Pinecone.
- **Conversational AI**: Uses NVIDIA's LLaMA 3.1 model to handle user queries with relevant context retrieval.
- **Vector Database Integration**: Efficient vector storage and retrieval using Pinecone.
- **Embedding Model**: Employs the `all-mpnet-base-v2` embedding model for creating high-quality vector representations.

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript (React)  
- **Backend**: Python (Flask)
- **Model Integration**: NVIDIA LLaMA 3.1 and Sentence Transformers (all-mpnet-base-v2)
- **Vector Database**: Pinecone


## Embedding Model

- **all-mpnet-base-v2**: 768 dimensions and uses dot product scoring instead of cosine


## Nvidia Model

- **nvidia / llama-3.1-nemotron-70b-instruct**: ![Llama-3.1-Nemotron-70B-Instruct](https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct) is a large language model customized by NVIDIA to improve the helpfulness of LLM generated responses to user queries.

### Model Architecture:
Architecture Type: Transformer
Network Architecture: Llama 3.1

### Input:
Input Type(s): Text
Input Format: String
Input Parameters: One Dimensional (1D)
Other Properties Related to Input: Max of 128k tokens

### Output:
Output Type(s): Text
Output Format: String
Output Parameters: One Dimensional (1D)
Other Properties Related to Output: Max of 4k tokens

---
