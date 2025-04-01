from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import requests
from pathlib import Path

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

# Initialize paths and variables
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HAMLET_FILE = DATA_DIR / "hamlet.txt"
CHUNKS_FILE = DATA_DIR / "hamlet_chunks.json"

# Load or download Hamlet text
@app.on_event("startup")
async def load_hamlet():
    global chunks
    
    # Download Hamlet if needed
    if not HAMLET_FILE.exists():
        print("Downloading Hamlet text...")
        url = "https://www.gutenberg.org/files/1524/1524-0.txt"
        response = requests.get(url)
        response.raise_for_status()
        
        text = response.text
        # Clean up the Project Gutenberg header/footer
        text = re.sub(r'^.*?ACT I\.', 'ACT I.', text, flags=re.DOTALL)
        text = re.sub(r'THE END.*$', 'THE END.', text, flags=re.DOTALL)
        
        with open(HAMLET_FILE, "w", encoding="utf-8") as f:
            f.write(text)
        print("Hamlet text downloaded and saved.")
    else:
        print("Loading existing Hamlet text file.")
        with open(HAMLET_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    
    # Use pre-chunked data or create basic chunks
    if CHUNKS_FILE.exists():
        print("Loading pre-chunked data...")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        print("Creating basic chunks...")
        chunks = chunk_by_scenes(text)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
    
    print(f"Loaded {len(chunks)} chunks from Hamlet")

# Function to chunk the text by scenes
def chunk_by_scenes(text, max_chunk_size=1500):
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

# Simple in-memory keyword search (no embeddings)
def keyword_search(query, chunks, top_k=3):
    # Convert query to lowercase and split into keywords
    keywords = query.lower().split()
    
    # Score each chunk by keyword matches
    chunk_scores = []
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(1 for keyword in keywords if keyword in chunk_lower)
        chunk_scores.append((idx, score))
    
    # Sort by score and get top_k
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunks[idx] for idx, score in chunk_scores[:top_k] if score > 0]
    
    # If no matches, return random chunks
    if not top_chunks:
        import random
        random_indices = random.sample(range(len(chunks)), min(top_k, len(chunks)))
        top_chunks = [chunks[idx] for idx in random_indices]
    
    return top_chunks

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Query model
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
async def query_hamlet(request: QueryRequest):
    global chunks
    
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Get relevant chunks using simple keyword search
        relevant_chunks = keyword_search(request.question, chunks)
        
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