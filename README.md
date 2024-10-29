# RAG Application: Retrieval-Augmented Generation with NVIDIA LLaMA 3.1, Pinecone, and MPNet

This project is a Retrieval-Augmented Generation (RAG) pipeline that leverages NVIDIA's **LLaMA 3.1** language model for natural language interaction, **Pinecone** for vector storage and retrieval, and the **all-mpnet-base-v2** model to create text embeddings. The application allows users to upload documents, process them into embeddings, store them in a vector database, and perform contextualized Q&A based on relevant retrieved documents.

## Features

- **Upload Document**: Supports .txt file uploads for document embedding and storage in Pinecone.
- **Conversational AI**: Uses NVIDIA's LLaMA 3.1 model to handle user queries with relevant context retrieval.
- **Vector Database Integration**: Efficient vector storage and retrieval using Pinecone.
- **Embedding Model**: Employs the `all-mpnet-base-v2` embedding model for creating high-quality vector representations.

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **Model Integration**: NVIDIA LLaMA 3.1 and Sentence Transformers (all-mpnet-base-v2)
- **Vector Database**: Pinecone

---
