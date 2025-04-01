import streamlit as st
import os
import sys
import nltk
from nltk.corpus import gutenberg
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Display basic system information
st.write("Python version:", sys.version)
st.write("Working directory:", os.getcwd())

# Try importing required packages
try:
    import chromadb
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer
    
    # Indicate successful imports
    st.success("All required packages imported successfully")
    
except ImportError as e:
    st.error(f"Error importing required packages: {e}")
    st.info("Please check that all dependencies are installed properly via requirements.txt")
    st.stop()

# Load environment variables (for local development)
load_dotenv()

# For Streamlit Cloud deployment, check for secrets
if 'OPENAI_API_KEY' not in os.environ:
    # Check if running on Streamlit Cloud
    try:
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    except Exception as e:
        st.error(f"OpenAI API key not found: {e}")
        st.write("Please add it to your .env file or Streamlit secrets.")
        st.stop()

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('gutenberg', quiet=True)
    return True

# Initialize NLTK
download_nltk_data()

# Load Hamlet text
@st.cache_data
def get_hamlet_text():
    return gutenberg.raw('shakespeare-hamlet.txt')

# Simple text chunking
@st.cache_data
def chunk_text(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Sentence transformer for embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Simple semantic search
def search_chunks(query, chunks, model, top_k=3):
    # Encode query and chunks
    query_embedding = model.encode([query])[0]
    chunk_embeddings = model.encode(chunks)
    
    # Calculate similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # Get top_k chunks
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# OpenAI client
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not api_key:
        st.error("OpenAI API key not found!")
        st.stop()
    return OpenAI(api_key=api_key)

# Generate answer
def generate_answer(query, contexts, client):
    prompt = f"""
    You are a helpful assistant answering questions about Shakespeare's Hamlet.
    Use the following context to answer the question:
    
    Context:
    {' '.join(contexts)}
    
    Question: {query}
    
    Answer:
    """
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7
    )
    
    return response.choices[0].text.strip()

# Streamlit UI
st.title("Shakespeare's Digital Wisdom")
st.markdown("### Ask questions about Hamlet and get AI-powered answers")

# Sidebar with app information
with st.sidebar:
    st.header("About this App")
    st.markdown("""
    This app uses Retrieval-Augmented Generation (RAG) to answer questions about Shakespeare's Hamlet.
    
    **Try asking questions like:**
    - What is Hamlet's relationship with Ophelia?
    - Explain the "To be or not to be" soliloquy.
    - What happens in Act 3, Scene 1?
    """)
    
    top_k = st.slider("Number of chunks to retrieve", 1, 5, 3)

# Load data and model
with st.spinner("Loading Hamlet text and model..."):
    hamlet_text = get_hamlet_text()
    chunks = chunk_text(hamlet_text)
    model = load_model()
    openai_client = get_openai_client()

# Main query interface
query = st.text_input("Ask a question about Hamlet:", 
                     placeholder="e.g., What is the main theme of Hamlet?")

if query:
    # Process query
    with st.spinner("Searching for relevant information..."):
        relevant_chunks = search_chunks(query, chunks, model, top_k=top_k)
    
    # Display chunks if wanted
    with st.expander("View relevant text chunks"):
        for i, chunk in enumerate(relevant_chunks, 1):
            st.markdown(f"**Chunk {i}**")
            st.text(chunk[:300] + "...")
    
    # Generate answer
    with st.spinner("Generating answer..."):
        answer = generate_answer(query, relevant_chunks, openai_client)
    
    # Display answer
    st.markdown("### Answer:")
    st.markdown(answer)

# Page configuration
st.set_page_config(
    page_title="Shakespeare's Digital Wisdom",
    page_icon="📚",
    layout="wide"
)

# Initialize session state for history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Function to initialize the RAG pipeline
@st.cache_resource
def initialize_rag_pipeline():
    """Initialize the RAG pipeline with caching."""
    # Download NLTK data if needed
    nltk.download('gutenberg', quiet=True)
    
    persist_path = "./hamlet_vector_db"
    collection_name = "hamlet_chunks"
    
    # Check if we already have a persisted database
    if os.path.exists(persist_path) and os.listdir(persist_path):
        st.write("Loading existing Hamlet database...")
        # Just load the existing data
        return RAGPipeline(
            collection_name=collection_name, 
            persist_path=persist_path,
            use_reranking=True
        )
    else:
        # Load Hamlet and create a new database
        st.write("First-time setup: Loading and processing Hamlet text...")
        hamlet_text = gutenberg.raw('shakespeare-hamlet.txt')
        
        # Create and initialize the RAG pipeline
        with st.spinner("Building knowledge base... (this may take a few minutes)"):
            rag = RAGPipeline(
                hamlet_text, 
                collection_name=collection_name,
                use_shakespeare_chunker=True,
                persist_path=persist_path,
                use_reranking=True
            )
        st.success("Knowledge base created successfully!")
        return rag

# UI Elements
st.title("Shakespeare's Digital Wisdom")
st.markdown("### Ask questions about Hamlet and get AI-powered answers")

# Sidebar with app information
with st.sidebar:
    st.header("About this App")
    st.markdown("""
    This app uses Retrieval-Augmented Generation (RAG) to answer questions about Shakespeare's Hamlet.
    
    **How it works:**
    1. The text of Hamlet is split into chunks
    2. Each chunk is converted to a vector embedding
    3. When you ask a question, the system finds relevant chunks
    4. An AI model generates an answer based on those chunks
    
    **Try asking questions like:**
    - What is Hamlet's relationship with Ophelia?
    - Explain the "To be or not to be" soliloquy.
    - What happens in Act 3, Scene 1?
    - Why does Hamlet hesitate to kill Claudius?
    """)
    
    # Add options for advanced users
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 10, 3)
    show_contexts = st.checkbox("Show retrieved contexts", value=False)
    
    # Add a button to clear conversation history
    if st.button("Clear Conversation History"):
        st.session_state.conversation_history = []
        st.success("Conversation history cleared!")

# Main content
try:
    # Initialize the RAG pipeline
    rag = initialize_rag_pipeline()
    
    # Chat interface
    query = st.text_input("Ask a question about Hamlet:", placeholder="e.g., What is the main theme of Hamlet?")
    
    if query:
        # Process the query
        with st.spinner("Thinking..."):
            result = rag.process_query(query, top_k=top_k)
        
        # Display the result
        answer = result["answer"]
        
        # Add to conversation history
        st.session_state.conversation_history.append({"question": query, "answer": answer})
        
        # Display retrieved context if enabled
        if show_contexts:
            with st.expander("Retrieved Context", expanded=True):
                for i, context in enumerate(result["formatted_contexts"], 1):
                    st.markdown(f"#### Context {i}")
                    st.text(context)
        
        # Display the answer
        st.markdown("### Answer:")
        st.markdown(answer)
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### Previous Questions and Answers")
        
        for i, exchange in enumerate(reversed(st.session_state.conversation_history)):
            if i > 0:  # Add separator between conversations
                st.markdown("---")
            
            st.markdown(f"**Q: {exchange['question']}**")
            st.markdown(exchange['answer'])

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("If this is your first time using the app, it may take a moment to initialize. Please refresh the page if needed.")

# After your imports, define the RAG classes directly in this file
class TextChunker:
    # Copy your TextChunker implementation here
    pass

class ShakespeareChunker:
    # Copy your ShakespeareChunker implementation here
    pass

# Copy the rest of your classes: VectorStore, LLMGenerator, RAGPipeline 

# Add the current directory to the path so we can import ophelia
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ophelia import RAGPipeline
except ImportError:
    st.error("Could not import RAGPipeline from ophelia.py. Check that the file is in the same directory.")
    st.stop() 