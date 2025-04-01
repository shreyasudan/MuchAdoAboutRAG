import streamlit as st
import nltk
import os
from nltk.corpus import gutenberg
from ophelia import RAGPipeline

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