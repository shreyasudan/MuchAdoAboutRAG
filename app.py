from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import json
import openai  # Use the module directly instead of the class
from dotenv import load_dotenv
import requests
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

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

# Function to chunk the text by scenes (fixed version)
def chunk_by_scenes(text, max_chunk_size=1500):
    # Make sure we have content
    if not text or len(text) < 100:
        print(f"Warning: Text is too short to chunk: {len(text)} characters")
        return [text]
    
    # Add debugging
    print(f"Text length: {len(text)} characters")
    
    # Split by acts/scenes with a more robust pattern
    scene_pattern = re.compile(r'(ACT [IVX]+\.[\s]*SCENE [IVX]+\.)', re.DOTALL | re.IGNORECASE)
    
    # Print the first match to verify pattern works
    first_match = scene_pattern.search(text)
    if first_match:
        print(f"First scene match: {first_match.group(0)}")
    else:
        print("No scene matches found! Using fallback chunking.")
        return simple_chunk(text, max_chunk_size)
    
    # Split the text
    scenes = scene_pattern.split(text)
    print(f"Split into {len(scenes)} segments")
    
    # Debug the split
    for i, segment in enumerate(scenes[:3]):  # Show first 3 segments
        print(f"Segment {i}: {segment[:50]}...")
    
    # Clean up and join headers with content
    chunks = []
    current_header = ""
    
    for i, section in enumerate(scenes):
        if scene_pattern.match(section):
            current_header = section.strip()
            print(f"Found header: {current_header}")
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
    
    # Fallback to simple chunking if no scenes were identified
    if not chunks:
        print("No chunks created with scene pattern. Using fallback chunking.")
        return simple_chunk(text, max_chunk_size)
    
    print(f"Created {len(chunks)} chunks with scene pattern")
    return chunks

# Simple fallback chunking function
def simple_chunk(text, max_chunk_size=1500):
    chunks = []
    
    # Split by double newlines to try to preserve paragraph structure
    paragraphs = text.split("\n\n")
    
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If still no chunks (extremely unlikely), force split by size
    if not chunks:
        for i in range(0, len(text), max_chunk_size):
            chunks.append(text[i:i+max_chunk_size])
    
    print(f"Created {len(chunks)} chunks with simple chunking")
    return chunks

# Load or download Hamlet text
@app.on_event("startup")
async def load_hamlet():
    global chunks
    
    # Download Hamlet if needed
    if not HAMLET_FILE.exists():
        print("Downloading Hamlet text...")
        try:
            url = "https://www.gutenberg.org/files/1524/1524-0.txt"
            response = requests.get(url)
            response.raise_for_status()
            
            text = response.text
            print(f"Downloaded {len(text)} characters")
            
            # Use a simple pattern to find the start of Act 1
            act1_match = re.search(r'ACT I\.', text)
            if act1_match:
                start_idx = act1_match.start()
                text = text[start_idx:]
                print(f"Trimmed to {len(text)} characters starting at 'ACT I.'")
            
            # Use a simple pattern to find the end
            end_match = re.search(r'THE END', text)
            if end_match:
                end_idx = end_match.end()
                text = text[:end_idx]
                print(f"Trimmed to {len(text)} characters ending at 'THE END'")
            
            with open(HAMLET_FILE, "w", encoding="utf-8") as f:
                f.write(text)
            print("Hamlet text downloaded and saved.")
        except Exception as e:
            print(f"Error downloading Hamlet: {e}")
            # Provide a minimal fallback text
            text = "ACT I. SCENE I. Hamlet is a tragedy by William Shakespeare."
            with open(HAMLET_FILE, "w", encoding="utf-8") as f:
                f.write(text)
    else:
        print("Loading existing Hamlet text file.")
        with open(HAMLET_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded {len(text)} characters from file")
    
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
    if len(chunks) == 0:
        print("WARNING: No chunks were created. Check the text format and chunking logic.")
        # Create at least one emergency chunk so the API doesn't fail completely
        chunks = ["Hamlet is a tragedy by William Shakespeare."]

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
        # Limit question length to prevent token issues
        question = request.question[:500]
        
        # Get relevant chunks using simple keyword search
        relevant_chunks = keyword_search(question, chunks)
        
        # Format the context for the LLM
        formatted_context = "\n\n".join(relevant_chunks)
        
        # Limit context to prevent token limit errors
        formatted_context = limit_context_tokens(formatted_context)
        
        # Generate the answer using older OpenAI API
        system_message = (
            "You are a helpful assistant that answers questions about Shakespeare's Hamlet. "
            "When referencing specific parts of the play, mention Act and Scene numbers. "
            "If you're unsure about something, say so rather than making up information. "
            "Keep your answers concise but informative."  # Added instruction for brevity
        )
        
        user_message = (
            f"Based on the following excerpts from Hamlet, answer this question concisely:\n\n"  # Added concisely
            f"QUESTION: {question}\n\n"
            f"EXCERPTS FROM HAMLET:\n{formatted_context}\n\n"
            f"Provide a thoughtful answer with proper citations to Act and Scene when relevant."
        )
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,  # Reduced from 600 to 500
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
        except Exception as openai_error:
            if "maximum context length" in str(openai_error):
                # If context is still too long, try with even less context
                shortened_context = limit_context_tokens(formatted_context, max_tokens=6000)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"QUESTION: {question}\n\nEXCERPTS FROM HAMLET (shortened):\n{shortened_context}"}
                    ],
                    max_tokens=400,
                    temperature=0.7
                )
                answer = response.choices[0].message.content
            else:
                raise
        
        return {"answer": answer}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if "maximum context length" in str(e):
            raise HTTPException(status_code=413, detail="The context is too large for this question. Please ask a more specific question.")
        else:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "chunks_loaded": len(chunks) if 'chunks' in globals() else 0}

# Adding a root endpoint for health checks
@app.get("/")
async def root():
    return {
        "status": "healthy", 
        "message": "Shakespeare's Digital Wisdom API is running. Use /query endpoint to ask questions.",
        "chunks_loaded": len(chunks) if 'chunks' in globals() else 0,
        "documentation": "/docs"
    }

# Add this function to your app.py
def limit_context_tokens(context, max_tokens=12000):
    """Limit the context to approximately max_tokens."""
    # Very rough estimation: 1 token ≈ 4 characters for English text
    chars_per_token = 4
    max_chars = max_tokens * chars_per_token
    
    if len(context) <= max_chars:
        return context
    
    # Try to cut at paragraph boundaries
    paragraphs = context.split("\n\n")
    limited_context = ""
    
    for paragraph in paragraphs:
        if len(limited_context) + len(paragraph) + 2 <= max_chars:
            limited_context += paragraph + "\n\n"
        else:
            break
    
    return limited_context

# Run the application with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 