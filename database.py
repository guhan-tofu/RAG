import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap  # Move start to create overlap
    return chunks

def process_txt_files_in_folder(folder_path, chunk_size=1000, chunk_overlap=200):
    all_chunks = []  # to store all file chunks if needed
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                # Split the text into chunks
                chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)  # if you want to store all chunks in a list
                # Optional: Save or process chunks here
                for i, chunk in enumerate(chunks):
                    print(f"Chunk {i+1} of {filename}:", chunk[:100], "...")  # preview the first 100 chars of each chunk
    return all_chunks

# Usage
folder_path = 'data'
all_chunks = process_txt_files_in_folder(folder_path)

print(type(all_chunks))
print(len(all_chunks))


embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device="cuda")


# Sentences are encoded/embedded by calling model.encode()
embeddings = embedding_model.encode(all_chunks,batch_size=32)
embeddings_dict = dict(zip(all_chunks, embeddings))

# See the embeddings
for sentence, embedding in embeddings_dict.items():
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

print(len(embeddings_dict))


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")



# initialize pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "ragdb"

dimension = 768
print("Dimension:",dimension)

if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",  # or "euclidean" based on your requirement
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)


# id, vector, metadata
pinecone_data = [
    (str(i), embedding, {"sentence": sentence})
    for i, (sentence, embedding) in enumerate(embeddings_dict.items())
]

# Upload embeddings in batches
batch_size = 32  # adjust batch size based on memory and latency requirements
for i in range(0, len(pinecone_data), batch_size):
    index.upsert(vectors=pinecone_data[i: i + batch_size])
