# imports
import nltk
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import os
from dotenv import load_dotenv


# download the gutenberg corpus
nltk.download('gutenberg')  # run once

# import the gutenberg corpus
from nltk.corpus import gutenberg

# load the text of Hamlet
hamlet_text = gutenberg.raw('shakespeare-hamlet.txt')
raw_text = gutenberg.raw('shakespeare-hamlet.txt')

# print the first 1000 characters of the text
print(raw_text[:1000])

# print the first 1000 characters of the text
print(raw_text[:1000])

# split (chunk) the text 
import re

def split_into_chunks(text, max_chars=500):
    # Clean up any weird whitespace/newlines
    text = text.replace('\n', ' ').strip()
    
    # Split by sentences using a regex or use nltk.sent_tokenize for more advanced splitting
    sentences = re.split(r'(?<=[.?!])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) <= max_chars:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            # Join current chunk into a single text block
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
    
    # Add the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

chunks = split_into_chunks(raw_text, max_chars=500)
print(f"Total chunks: {len(chunks)}")
print(chunks[0])  # See what the first chunk looks like

# Embedder

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # or any other sentence-transformers model
embeddings = model.encode(chunks, show_progress_bar=True)

# Store Embeddings in a Vector Database (Chroma)

# Create a client (by default, it uses an in-memory database)
client = chromadb.Client(Settings(
    persist_directory="."  # or some path to store embeddings on disk
))

collection = client.create_collection("hamlet_chunks")

# Add your chunks + embeddings to the collection
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]  # unique IDs for each chunk
)

# Query the collection

def retrieve_relevant_chunks(query, top_k=3):
    # Embed the query
    query_embedding = model.encode([query])
    
    # Query Chroma for the top-k similar chunks
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )
    
    # results['documents'] is a list of lists; each sublist has the top docs
    relevant_chunks = results['documents'][0]
    
    return relevant_chunks

user_question = "What is the main theme of Hamlet's soliloquy?"
retrieved_chunks = retrieve_relevant_chunks(user_question, top_k=3)
print("Retrieved Chunks:\n")
for idx, chunk in enumerate(retrieved_chunks, start=1):
    print(f"Chunk #{idx}:\n{chunk}\n")

# RAG Pipeline

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_answer(question, context):
    system_prompt = (
        "You are a helpful assistant that uses the text provided to answer questions.\n"
        "If you don't have enough information, say so. If you do, answer succinctly.\n\n"
    )
    # Combine all retrieved chunks
    context_text = "\n\n".join(context)
    
    prompt = (
        f"{system_prompt}"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  # text-davinci-003 is deprecated
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    
    return response.choices[0].text.strip()

# Use the retrieved chunks
answer = generate_answer(user_question, retrieved_chunks)
print("Answer:", answer)
