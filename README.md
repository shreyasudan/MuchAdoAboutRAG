# Shakespeare's Digital Wisdom: A RAG Implementation

This project demonstrates a Retrieval-Augmented Generation (RAG) system that allows you to ask questions about Shakespeare's "Hamlet" and receive contextually relevant answers powered by AI.

## What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models by retrieving relevant information from a knowledge base before generating responses. This approach:

1. **Grounds the AI in factual information** (in this case, the text of Hamlet)
2. **Reduces hallucinations** by providing specific context
3. **Enables domain-specific knowledge** without fine-tuning the model

## Project Components
- `muchAdo.py`: A script-style implementation of the RAG pipeline
- `ophelia.py`: An object-oriented implementation with interactive query capabilities

## Getting Started

### Prerequisites
- Python 3.8+
- An OpenAI API Key

### Installation
1. Clone this repository:
```python
git clone https://github.com/shreyasudan/MuchAdoAboutRAG.git
cd shakespeare-rag
```
2. Create a virtual environment and activate it:
```python
python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```
3. Install required packages:
```python
pip install nltk chromadb openai python-dotenv sentence-transformers
```
### Running the interavtive RAG System
Try the object-oriented interactive version:
```python
python ophelia.py
```
This will:
1. Download the text of Hamlet
2. Split it into manageable chunks
3. Create embeddings for each chunk
4. Store them in a vector database
5. Launch an interactive console where you can ask questions

Example questions to try:
1. "What is Hamlet's relationship with Ophelia?"
2. "Explain the 'To be or not to be' soliloquy."
3. "What are the main themes of the play?"

## How it Works
1. **Text Chunking**: The full text of Hamlet is split into semantically meaningful chunks.
2. **Embedding**: Each chunk is converted into a vector representation using SentenceTransformers.
3. **Storage**: The embeddings are stored in a ChromaDB vector database.
4. **Retrieval**: When you ask a question, the system finds the most semantically similar chunks.
5. **Generation**: The relevant chunks are sent to OpenAI's API as context, along with your question.
