from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
import os
from nltk.corpus import gutenberg
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Shakespeare's Digital Wisdom API",
    description="An API for answering questions about Shakespeare's Hamlet using RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your GitHub Pages domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download NLTK data on startup
@app.on_event("startup")
async def startup_event():
    nltk.download('gutenberg', quiet=True)
    print("NLTK data downloaded successfully")

# Load Hamlet text and prepare chunks
@app.on_event("startup")
async def load_hamlet():
    global chunks, chunk_embeddings
    
    # Load the text
    hamlet_text = gutenberg.raw('shakespeare-hamlet.txt')
    
    # Advanced chunking by scenes or structure
    chunks = chunk_by_scenes(hamlet_text)
    
    # Initialize the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate embeddings for all chunks
    chunk_embeddings = model.encode(chunks, show_progress_bar=True)
    
    print(f"Loaded {len(chunks)} chunks from Hamlet")

# Function to chunk the text by scenes or dramatic structure
def chunk_by_scenes(text, max_chunk_size=1500):
    # Remove the initial metadata
    text = re.sub(r'^.*?ACT I\.', 'ACT I.', text, flags=re.DOTALL)
    
    # Split by acts/scenes
    scene_pattern = re.compile(r'(ACT [IVX]+\.\s+SCENE [IVX]+\.)', re.DOTALL)
    scenes = scene_pattern.split(text)
    
    # Clean up and join headers with content
    chunks = []
    current_header = ""
    
    for i, section in enumerate(scenes):
        if scene_pattern.match(section):
            current_header = section.strip()
        elif section.strip() and current_header:
            # Chunk large scenes further if needed
            scene_text = section
            if len(scene_text) > max_chunk_size:
                # Use a sliding window if the scene is too large
                for j in range(0, len(scene_text), max_chunk_size // 2):
                    end_idx = min(j + max_chunk_size, len(scene_text))
                    sub_chunk = scene_text[j:end_idx]
                    chunks.append(f"{current_header}\n\n{sub_chunk}")
            else:
                chunks.append(f"{current_header}\n\n{scene_text}")
    
    return chunks

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Query model
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
async def query_hamlet(request: QueryRequest):
    global chunks, chunk_embeddings
    
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Encode the query
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode([request.question])[0]
        
        # Calculate similarity with all chunks
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Get top 3 most relevant chunks
        top_k = 3
        top_indices = similarities.argsort()[-top_k:][::-1]
        relevant_chunks = [chunks[i] for i in top_indices]
        
        # Format the context for the LLM
        formatted_context = "\n\n".join(relevant_chunks)
        
        # Generate the answer
        system_prompt = (
            "You are a helpful assistant that answers questions about Shakespeare's Hamlet. "
            "When referencing specific parts of the play, mention Act and Scene numbers. "
            "If you're unsure about something, say so rather than making up information. "
            "Use formal language appropriate for discussing Shakespeare."
        )
        
        user_prompt = (
            f"Based on the following excerpts from Hamlet, answer the question:\n\n"
            f"QUESTION: {request.question}\n\n"
            f"EXCERPTS FROM HAMLET:\n{formatted_context}\n\n"
            f"Provide a thoughtful answer with proper citations to Act and Scene when relevant."
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        return {"answer": answer}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "chunks_loaded": len(chunks) if 'chunks' in globals() else 0}

# Run the application with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 