import os
from getpass import getpass
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Set NVIDIA API key
if not os.getenv("NVIDIA_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = getpass("Enter your NVIDIA API key: ")

# Initialize the model
MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
model = ChatNVIDIA(model_name=MODEL)

# Initialize conversation history
conversation_history = []



# Initialize embedding model to convert query
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device="cuda")


# Open pinecone index to perform similarty search
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "ragdb"
index = pc.Index(index_name)



def retrieve_document(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    response = index.query(vector=query_embedding,top_k=top_k, include_metadata=True) # include metadata because it contains the actual text
    return [match['metadata']['sentence'] for match in response['matches']]


# Function to interact with the model while preserving context
def chat_with_model(user_input):
    # Append the user input to the conversation history
    relevant_docs = retrieve_document(user_input)
    augmented_input = f"""You are a helpful assistant, below is a query from a user and
    some relevant contexts. Answer the question given the information in those
    contexts. If you cannot find the answer to the question, say "I don't know".

    Contexts:
    {relevant_docs}

    Query: {user_input}

    Answer: """
    conversation_history.append({"role": "user", "content": augmented_input})
    # Make the request to the model with the conversation history
    response = model.invoke(conversation_history)
    conversation_history.pop()
    conversation_history.append({"role": "user", "content": user_input})

    
    # Append the model's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response.content})
    
    # Return the response content
    return response.content

# Example conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    response = chat_with_model(user_input)
    print("Assistant:", response)
