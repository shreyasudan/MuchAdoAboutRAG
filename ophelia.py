import nltk
import chromadb
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from nltk.corpus import gutenberg

class TextChunker:
    def __init__(self, max_chars=500):
        self.max_chars = max_chars
    
    def chunk(self, text):
        # Clean up any weird whitespace/newlines
        text = text.replace('\n', ' ').strip()
        
        # Split by sentences using a regex
        sentences = re.split(r'(?<=[.?!])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) <= self.max_chars:
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

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_path="."):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client(Settings(persist_directory=persist_path))
        self.collection = None
    
    def create_collection(self, collection_name):
        self.collection = self.client.create_collection(collection_name)
        return self.collection
    
    def get_collection(self, collection_name):
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.create_collection(collection_name)
        return self.collection
    
    def index_documents(self, chunks, collection_name="document_chunks"):
        self.get_collection(collection_name)
        
        # Generate embeddings
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        # Add to collection
        self.collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        
        print(f"Indexed {len(chunks)} chunks into collection '{collection_name}'")
    
    def retrieve(self, query, top_k=3):
        if not self.collection:
            raise ValueError("No collection selected. Call get_collection() first.")
        
        # Embed the query
        query_embedding = self.model.encode([query])
        
        # Query for similar chunks
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Return relevant chunks
        return results['documents'][0]

class LLMGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, query, context, model="gpt-3.5-turbo-instruct"):
        system_prompt = (
            "You are a helpful assistant that uses the text provided to answer questions.\n"
            "If you don't have enough information, say so. If you do, answer succinctly.\n\n"
        )
        
        # Combine all context chunks
        context_text = "\n\n".join(context)
        
        prompt = (
            f"{system_prompt}"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].text.strip()

class ShakespeareChunker:
    def __init__(self):
        self.act_pattern = re.compile(r'ACT [IVX]+\.', re.IGNORECASE)
        self.scene_pattern = re.compile(r'SCENE [IVX]+\.', re.IGNORECASE)
        self.character_pattern = re.compile(r'([A-Z]{2,})\.')
        
    def extract_structure(self, text):
        """Extract the structural elements of the play."""
        # Remove starting metadata about the play
        cleaned_text = re.sub(r'^.*?ACT I\.', 'ACT I.', text, flags=re.DOTALL)
        
        # Split by acts
        acts = self.act_pattern.split(cleaned_text)
        acts = [f"ACT {i+1}. {act}" for i, act in enumerate(acts[1:]) if act.strip()]
        
        structured_play = []
        
        for act in acts:
            # Split by scenes
            scenes = self.scene_pattern.split(act)
            act_num = scenes[0].strip()
            scenes = [f"{act_num} SCENE {i+1}. {scene}" for i, scene in enumerate(scenes[1:]) if scene.strip()]
            
            for scene in scenes:
                # Extract character dialogues and context
                scene_parts = []
                
                # Get the scene header (location and setup)
                header_match = re.search(r'^.*?\n\n', scene, re.DOTALL)
                if header_match:
                    scene_header = header_match.group(0).strip()
                    scene_parts.append({
                        "type": "scene_setting",
                        "content": scene_header
                    })
                
                # Split dialogue and extract character speeches
                scene_body = scene[header_match.end():] if header_match else scene
                dialogues = re.split(r'\n\s*\n', scene_body)
                
                for dialogue in dialogues:
                    # Check if this is a character's speech
                    char_match = self.character_pattern.match(dialogue)
                    if char_match:
                        character = char_match.group(1)
                        speech = dialogue[char_match.end():].strip()
                        scene_parts.append({
                            "type": "dialogue",
                            "character": character,
                            "content": speech
                        })
                    else:
                        # This is stage direction or other text
                        if dialogue.strip():
                            scene_parts.append({
                                "type": "direction",
                                "content": dialogue.strip()
                            })
                
                structured_play.append({
                    "type": "scene",
                    "header": scene.split('\n', 1)[0].strip(),
                    "parts": scene_parts
                })
        
        return structured_play
    
    def chunk_by_structure(self, text, max_chunk_size=1000):
        """Split the text by its dramatic structure rather than arbitrary lengths."""
        structured_play = self.extract_structure(text)
        chunks = []
        
        for scene in structured_play:
            # Create identifier for the scene
            scene_id = scene["header"]
            
            # Start with scene setting
            current_chunk = f"{scene_id}\n\n"
            current_length = len(current_chunk)
            
            for part in scene["parts"]:
                part_text = ""
                if part["type"] == "scene_setting":
                    part_text = f"Setting: {part['content']}\n\n"
                elif part["type"] == "dialogue":
                    part_text = f"{part['character']}: {part['content']}\n\n"
                elif part["type"] == "direction":
                    part_text = f"[Direction: {part['content']}]\n\n"
                
                # Check if adding this part would exceed max chunk size
                if current_length + len(part_text) > max_chunk_size:
                    # Save current chunk and start a new one
                    chunks.append(current_chunk.strip())
                    current_chunk = f"(Continued) {scene_id}\n\n{part_text}"
                    current_length = len(current_chunk)
                else:
                    current_chunk += part_text
                    current_length += len(part_text)
            
            # Add the final chunk from this scene
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk(self, text, max_chunk_size=1000):
        """Main interface method that returns chunks of the play."""
        # First try structural chunking
        chunks = self.chunk_by_structure(text, max_chunk_size)
        
        # If structural chunking failed or produced too few chunks, fallback to basic chunking
        if not chunks or len(chunks) < 5:
            print("Structural chunking failed, falling back to basic chunking")
            basic_chunker = TextChunker(max_chunk_size)
            chunks = basic_chunker.chunk(text)
        
        return chunks

class RAGPipeline:
    def __init__(self, source_text=None, collection_name="hamlet_chunks", use_shakespeare_chunker=True):
        # Use Shakespeare-specific chunker if requested
        self.chunker = ShakespeareChunker() if use_shakespeare_chunker else TextChunker()
        self.vector_store = VectorStore()
        self.generator = LLMGenerator()
        self.collection_name = collection_name
        
        # If source text is provided, initialize the system
        if source_text:
            self.initialize(source_text)
    
    def initialize(self, source_text):
        print("Chunking text...")
        chunks = self.chunker.chunk(source_text)
        print(f"Created {len(chunks)} chunks")
        
        print("Indexing chunks...")
        self.vector_store.index_documents(chunks, self.collection_name)
        print("Indexing complete")
    
    def process_query(self, query, top_k=3):
        print(f"Processing query: '{query}'")
        
        # Get the collection
        self.vector_store.get_collection(self.collection_name)
        
        # Retrieve relevant chunks
        print(f"Retrieving top {top_k} relevant chunks...")
        chunks = self.vector_store.retrieve(query, top_k)
        
        # Generate answer
        print("Generating answer...")
        answer = self.generator.generate(query, chunks)
        
        return {
            "query": query,
            "chunks": chunks,
            "answer": answer
        }
    
    def interactive_mode(self):
        print("\n===== RAG Interactive Mode =====")
        print("Type your questions about Hamlet (or 'exit' to quit)")
        
        while True:
            query = input("\nYour question: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            try:
                result = self.process_query(query)
                
                print("\n----- Retrieved Context -----")
                for i, chunk in enumerate(result["chunks"], 1):
                    print(f"Chunk {i}: {chunk[:100]}...")
                
                print("\n----- Answer -----")
                print(result["answer"])
                
            except Exception as e:
                print(f"Error: {e}")


def main():
    # Download NLTK data if needed
    nltk.download('gutenberg')
    
    # Load Hamlet
    print("Loading Hamlet text...")
    hamlet_text = gutenberg.raw('shakespeare-hamlet.txt')
    
    # Create and initialize the RAG pipeline with Shakespeare-specific chunking
    print("Initializing RAG pipeline with Shakespeare-specific text chunking...")
    rag = RAGPipeline(hamlet_text, use_shakespeare_chunker=True)
    
    # Start interactive mode
    rag.interactive_mode()


if __name__ == "__main__":
    main()
