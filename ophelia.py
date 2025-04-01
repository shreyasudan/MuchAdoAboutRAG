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
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_path="./vector_db"):
        self.model = SentenceTransformer(model_name)
        self.persist_path = persist_path
        
        # Create directory if it doesn't exist
        os.makedirs(persist_path, exist_ok=True)
        
        # Initialize the client with persistence
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = None
        
        # Initialize embedding cache
        self.embedding_cache = {}
        self.model_name = model_name
        
        # Load cached embeddings if available
        self._load_embedding_cache()
        
        # For re-ranking
        self.cross_encoder = None
    
    def _cache_path(self):
        """Get the path for the embedding cache file."""
        return os.path.join(self.persist_path, f"embedding_cache_{self.model_name.replace('/', '_')}.pkl")
    
    def _load_embedding_cache(self):
        """Load embedding cache from disk if it exists."""
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Failed to load embedding cache: {e}")
                self.embedding_cache = {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        cache_path = self._cache_path()
        try:
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            print(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            print(f"Failed to save embedding cache: {e}")
    
    def get_embedding(self, text):
        """Get embedding with caching."""
        # Use a hash of the text as the cache key
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Generate new embedding
        embedding = self.model.encode(text)
        
        # Cache it
        self.embedding_cache[text_hash] = embedding
        
        # Save periodically (every 100 new embeddings)
        if len(self.embedding_cache) % 100 == 0:
            self._save_embedding_cache()
            
        return embedding
    
    def create_collection(self, collection_name):
        """Create a new collection."""
        # Check if collection exists first
        try:
            return self.client.get_collection(collection_name)
        except:
            return self.client.create_collection(collection_name)
    
    def get_collection(self, collection_name):
        """Get or create a collection."""
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.create_collection(collection_name)
        return self.collection
    
    def index_documents(self, chunks, metadata=None, collection_name="document_chunks"):
        """Index documents with metadata and cached embeddings."""
        print(f"Indexing {len(chunks)} documents into collection '{collection_name}'...")
        self.get_collection(collection_name)
        
        # Check if documents are already indexed
        existing_ids = set()
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                print(f"Found {existing_count} existing documents in collection")
                # Get existing IDs
                existing_ids = set(self.collection.get()["ids"])
        except Exception as e:
            print(f"Error checking existing documents: {e}")
        
        # Generate new IDs for chunks
        doc_ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Find which chunks need to be indexed
        new_indices = [i for i, doc_id in enumerate(doc_ids) if doc_id not in existing_ids]
        
        if not new_indices:
            print("All documents already indexed.")
            return
        
        new_chunks = [chunks[i] for i in new_indices]
        new_ids = [doc_ids[i] for i in new_indices]
        
        # Include metadata if provided
        new_metadata = None
        if metadata:
            new_metadata = [metadata[i] for i in new_indices]
        
        print(f"Generating embeddings for {len(new_chunks)} new documents...")
        
        # Generate embeddings with cache
        embeddings = []
        for chunk in new_chunks:
            embedding = self.get_embedding(chunk)
            embeddings.append(embedding.tolist())
        
        # Add to collection with metadata
        if new_metadata:
            self.collection.add(
                documents=new_chunks,
                embeddings=embeddings,
                metadatas=new_metadata,
                ids=new_ids
            )
        else:
            self.collection.add(
                documents=new_chunks,
                embeddings=embeddings,
                ids=new_ids
            )
        
        # Save cache after indexing
        self._save_embedding_cache()
        
        print(f"Indexed {len(new_chunks)} new documents into collection '{collection_name}'")
    
    def _load_cross_encoder(self):
        """Load cross-encoder model for re-ranking."""
        if self.cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                print("Loading cross-encoder model for re-ranking...")
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("Cross-encoder model loaded.")
            except Exception as e:
                print(f"Failed to load cross-encoder: {e}")
                print("Continuing without re-ranking.")
    
    def retrieve(self, query, top_k=3, rerank=True, rerank_top_k=10):
        """Retrieve relevant documents with optional re-ranking."""
        if not self.collection:
            raise ValueError("No collection selected. Call get_collection() first.")
        
        # Embed the query
        query_embedding = self.model.encode([query])
        
        # First retrieval phase - get more documents than needed for re-ranking
        k_retrieval = rerank_top_k if rerank else top_k
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k_retrieval,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = results['documents'][0]
        metadatas = results.get('metadatas', [[None] * len(documents)])[0]
        distances = results.get('distances', [[0] * len(documents)])[0]
        
        # If re-ranking is enabled and we have documents to re-rank
        if rerank and len(documents) > top_k:
            try:
                self._load_cross_encoder()
                if self.cross_encoder:
                    # Prepare pairs for cross-encoder
                    pairs = [[query, doc] for doc in documents]
                    
                    # Get cross-encoder scores
                    print("Re-ranking with cross-encoder...")
                    scores = self.cross_encoder.predict(pairs)
                    
                    # Sort documents by cross-encoder score
                    scored_results = list(zip(documents, metadatas, scores))
                    scored_results.sort(key=lambda x: x[2], reverse=True)
                    
                    # Take top_k after re-ranking
                    documents = [item[0] for item in scored_results[:top_k]]
                    metadatas = [item[1] for item in scored_results[:top_k]]
                    
                    print(f"Re-ranked to top {top_k} documents")
            except Exception as e:
                print(f"Re-ranking failed: {e}")
                print("Using original ranking")
                # Just take the top_k from the original results
                documents = documents[:top_k]
                metadatas = metadatas[:top_k]
        elif len(documents) > top_k:
            # Limit to top_k if we're not re-ranking
            documents = documents[:top_k]
            metadatas = metadatas[:top_k]
        
        return {"documents": documents, "metadatas": metadatas}

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
            "If you don't have enough information, say so. If you do, answer comprehensively.\n\n"
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
            max_tokens=500,
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
        """Split the text by its dramatic structure and include metadata."""
        structured_play = self.extract_structure(text)
        chunks = []
        chunk_metadata = []
        
        for scene_idx, scene in enumerate(structured_play):
            # Parse act and scene numbers
            scene_header = scene["header"]
            act_match = re.search(r'ACT ([IVX]+)', scene_header)
            scene_match = re.search(r'SCENE ([IVX]+)', scene_header)
            
            act_num = act_match.group(1) if act_match else "Unknown"
            scene_num = scene_match.group(1) if scene_match else "Unknown"
            
            # Create identifier for the scene
            scene_id = scene["header"]
            
            # Start with scene setting
            current_chunk = f"{scene_id}\n\n"
            current_length = len(current_chunk)
            
            # Track which characters are in this chunk
            current_characters = set()
            chunk_start_idx = 0
            
            for part_idx, part in scene["parts"]:
                part_text = ""
                if part["type"] == "scene_setting":
                    part_text = f"Setting: {part['content']}\n\n"
                elif part["type"] == "dialogue":
                    character = part['character']
                    current_characters.add(character)
                    part_text = f"{character}: {part['content']}\n\n"
                elif part["type"] == "direction":
                    part_text = f"[Direction: {part['content']}]\n\n"
                
                # Check if adding this part would exceed max chunk size
                if current_length + len(part_text) > max_chunk_size:
                    # Save current chunk with its metadata
                    chunks.append(current_chunk.strip())
                    
                    # Create metadata for this chunk
                    metadata = {
                        "act": act_num,
                        "scene": scene_num,
                        "scene_id": scene_id,
                        "characters": list(current_characters),
                        "chunk_type": "scene_continuation" if chunk_start_idx > 0 else "scene_start",
                        "scene_index": scene_idx,
                        "chunk_index": chunk_start_idx
                    }
                    chunk_metadata.append(metadata)
                    
                    # Start a new chunk
                    current_chunk = f"(Continued) {scene_id}\n\n{part_text}"
                    current_length = len(current_chunk)
                    current_characters = {character} if part["type"] == "dialogue" else set()
                    chunk_start_idx += 1
                else:
                    current_chunk += part_text
                    current_length += len(part_text)
            
            # Add the final chunk from this scene with its metadata
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                metadata = {
                    "act": act_num,
                    "scene": scene_num,
                    "scene_id": scene_id,
                    "characters": list(current_characters),
                    "chunk_type": "scene_continuation" if chunk_start_idx > 0 else "scene_complete",
                    "scene_index": scene_idx,
                    "chunk_index": chunk_start_idx
                }
                chunk_metadata.append(metadata)
        
        return chunks, chunk_metadata
    
    def chunk(self, text, max_chunk_size=1000):
        """Main interface method that returns chunks of the play with metadata."""
        # First try structural chunking
        chunks, metadata = self.chunk_by_structure(text, max_chunk_size)
        
        # If structural chunking failed, fallback to basic chunking
        if not chunks or len(chunks) < 5:
            print("Structural chunking failed, falling back to basic chunking")
            basic_chunker = TextChunker(max_chunk_size)
            chunks = basic_chunker.chunk(text)
            # Create basic metadata
            metadata = [{"chunk_index": i, "chunk_type": "basic"} for i in range(len(chunks))]
        
        return chunks, metadata

class RAGPipeline:
    def __init__(self, source_text=None, collection_name="hamlet_chunks", use_shakespeare_chunker=True, 
                 persist_path="./vector_db"):
        # Use Shakespeare-specific chunker if requested
        self.chunker = ShakespeareChunker() if use_shakespeare_chunker else TextChunker()
        self.vector_store = VectorStore(persist_path=persist_path)
        self.generator = LLMGenerator()
        self.collection_name = collection_name
        
        # If source text is provided, initialize the system
        if source_text:
            self.initialize(source_text)
        else:
            # Try to load existing collection
            try:
                self.vector_store.get_collection(self.collection_name)
                print(f"Successfully loaded existing collection '{self.collection_name}'")
            except Exception as e:
                print(f"Failed to load collection: {e}")
    
    def initialize(self, source_text):
        print("Chunking text...")
        chunks, metadata = self.chunker.chunk(source_text)
        print(f"Created {len(chunks)} chunks")
        
        print("Indexing chunks...")
        self.vector_store.index_documents(chunks, metadata, self.collection_name)
        print("Indexing complete")
    
    def process_query(self, query, top_k=3):
        print(f"Processing query: '{query}'")
        
        # Get the collection
        self.vector_store.get_collection(self.collection_name)
        
        # Retrieve relevant chunks
        print(f"Retrieving top {top_k} relevant chunks...")
        results = self.vector_store.retrieve(query, top_k)
        
        # Generate answer
        print("Generating answer...")
        answer = self.generator.generate(query, results["documents"])
        
        return {
            "query": query,
            "results": results,
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
                for i, doc in enumerate(result["results"]["documents"], 1):
                    # Show a snippet of each document (first 100 chars)
                    print(f"Document {i}: {doc[:100].replace('\n', ' ')}...")
                
                print("\n----- Answer -----")
                # Print the full answer
                print(result["answer"])
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    # Download NLTK data if needed
    nltk.download('gutenberg')
    
    persist_path = "./hamlet_vector_db"
    collection_name = "hamlet_chunks"
    
    # Check if we already have a persisted database
    if os.path.exists(persist_path) and os.listdir(persist_path):
        print(f"Found existing vector database at {persist_path}")
        # Just load the existing data
        rag = RAGPipeline(collection_name=collection_name, persist_path=persist_path)
    else:
        # Load Hamlet and create a new database
        print("Loading Hamlet text...")
        hamlet_text = gutenberg.raw('shakespeare-hamlet.txt')
        
        # Create and initialize the RAG pipeline
        print("Initializing RAG pipeline with Shakespeare-specific text chunking...")
        rag = RAGPipeline(
            hamlet_text, 
            collection_name=collection_name,
            use_shakespeare_chunker=True,
            persist_path=persist_path
        )
    
    # Start interactive mode
    rag.interactive_mode()


if __name__ == "__main__":
    main()
