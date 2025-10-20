import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2
from typing import List, Dict
import tempfile
import glob
from pathlib import Path
import hashlib

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self, documents_folder="./documents"):
        # Initialize Google Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

        # Documents folder
        self.documents_folder = documents_folder
        self.processed_files = self.get_processed_files()

    def get_processed_files(self) -> set:
        """Get list of already processed files from ChromaDB metadata"""
        try:
            all_data = self.collection.get()
            processed = set()
            for metadata in all_data['metadatas']:
                if 'source' in metadata:
                    processed.add(metadata['source'])
            return processed
        except:
            return set()

    def get_file_hash(self, filepath: str) -> str:
        """Generate hash of file content to detect changes"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""

    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            st.error(f"Error reading TXT {txt_path}: {str(e)}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def scan_documents_folder(self) -> List[str]:
        """Scan documents folder for PDF and TXT files"""
        if not os.path.exists(self.documents_folder):
            os.makedirs(self.documents_folder)
            return []

        files = []
        # Get PDF files
        files.extend(glob.glob(os.path.join(self.documents_folder, "*.pdf")))
        # Get TXT files
        files.extend(glob.glob(os.path.join(self.documents_folder, "*.txt")))

        return files

    def auto_index_documents(self, progress_callback=None):
        """Automatically index all documents in the documents folder"""
        files = self.scan_documents_folder()

        if not files:
            if progress_callback:
                progress_callback("📁 No documents found in ./documents folder")
            return []

        new_files = []
        updated_files = []

        for file_path in files:
            filename = os.path.basename(file_path)
            file_hash = self.get_file_hash(file_path)

            # Check if file is new or updated
            stored_hash = self.get_stored_file_hash(filename)

            if filename not in self.processed_files or stored_hash != file_hash:
                if progress_callback:
                    progress_callback(f"📄 Processing: {filename}")

                # Extract text based on file type
                if file_path.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                else:
                    text = self.extract_text_from_txt(file_path)

                if text:
                    # Remove old version if exists
                    if filename in self.processed_files:
                        self.remove_document(filename)
                        updated_files.append(filename)
                    else:
                        new_files.append(filename)

                    # Add new version
                    self.add_document(text, filename, file_hash)
                    self.processed_files.add(filename)

        return new_files, updated_files

    def get_stored_file_hash(self, filename: str) -> str:
        """Get stored hash for a file"""
        try:
            results = self.collection.get(
                where={"source": filename, "chunk_id": 0}
            )
            if results['metadatas']:
                return results['metadatas'][0].get('file_hash', '')
        except:
            pass
        return ""

    def remove_document(self, filename: str):
        """Remove all chunks of a document from the database"""
        try:
            # Get all IDs for this document
            results = self.collection.get(where={"source": filename})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        except Exception as e:
            st.error(f"Error removing document {filename}: {str(e)}")

    def add_document(self, text: str, filename: str, file_hash: str = ""):
        """Add document to vector database"""
        chunks = self.chunk_text(text)

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks).tolist()

        # Create IDs and metadata
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_id": i, "file_hash": file_hash} for i in range(len(chunks))]

        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

    def search_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for relevant document chunks with fuzzy matching fallback"""
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        # If no good results found, try fuzzy matching by checking for typos
        if not results['documents'][0] or self._is_low_similarity(results):
            fuzzy_matches = self._fuzzy_search(query, n_results)
            if fuzzy_matches:
                return fuzzy_matches

        return results

    def _is_low_similarity(self, results: Dict) -> bool:
        """Check if results have low similarity scores"""
        if results['documents'][0]:
            # Check distances - higher distance means lower similarity
            distances = results.get('distances', [[1.0]])
            return all(d > 0.3 for d in distances[0])  # Threshold for low similarity
        return True

    def _fuzzy_search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Fallback fuzzy search for potential typos"""
        try:
            # Get all documents for fuzzy matching
            all_docs = self.collection.get()

            if not all_docs['documents']:
                return None

            # Simple fuzzy matching by finding similar terms in documents
            query_words = query.lower().split()
            best_matches = []
            best_scores = []

            for i, doc in enumerate(all_docs['documents']):
                doc_text = doc.lower()
                score = 0

                # Check for similar words or partial matches
                for word in query_words:
                    if len(word) > 3:  # Only check longer words for typos
                        # Look for words that contain most of the query word
                        words_in_doc = doc_text.split()
                        for doc_word in words_in_doc:
                            # Simple similarity check
                            if len(word) > 3 and len(doc_word) > 3:
                                # Check if words share many characters
                                common_chars = set(word) & set(doc_word)
                                total_chars = set(word) | set(doc_word)
                                if len(common_chars) / len(total_chars) > 0.7:  # 70% similarity
                                    score += 1

                if score > 0:
                    best_matches.append({
                        'documents': [[doc]],
                        'metadatas': [[all_docs['metadatas'][i]]],
                        'distances': [[0.2]],  # Lower distance for fuzzy matches
                        'ids': [[all_docs['ids'][i]]]
                    })

                if len(best_matches) >= n_results:
                    break

            return best_matches[0] if best_matches else None

        except Exception as e:
            print(f"Fuzzy search error: {e}")
            return None

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Gemini with strict document adherence"""
        prompt = f"""
    You are a highly reliable assistant that answers ONLY using the provided context from the following four manuals. 

    STRICT RULES:
    1. ONLY use information found in the provided context below—do NOT make up information or rely on external sources.
    2. If the answer is not present in the context, reply: "I don't have information about that in the provided documents."
    3. Always cite which document the information is sourced from, using its filename.
    4. Be concise, accurate, and clear. Structure your responses for easy understanding and quick reference.
    5. If multiple documents provide relevant information, synthesize the answer and cite all applicable sources.
    6. Avoid any assumptions or guesses—stick strictly to the facts in the context.
    7. 'Hey' or 'Hello' type greetings should be responded with simple acknowledgment like "Hello! How can I assist you today?" or similar.
    8. If the query is vague or too broad, ask for clarification rather than guessing.
    9. If the query is unrelated to the documents, politely inform the user that you can only assist with topics covered in the provided manuals.
    10. Maintain a professional and neutral tone throughout.

    Below are detailed context summaries extracted from the four provided manuals:

    -----
    [User_Manual_for_Vendors_on_Post_Contract_Activities_-Version-2.0.pdf]
    Purpose: Guidance for suppliers on post-contract processes in IREPS—handling digital challans, receipt notes, consignment receipts, bill submission, and supplementary bills.
    Main flows: View/download documents; submit digital bills; attach supporting documents; digitally sign; track bill/payment status; manage supplementary claims.

    -----
    [User-Manual-for-Contractors-for-E-Auction-for-Earning-Leasing-Contracts-Version-1.0.pdf]
    Purpose: Contractors' guide for asset leasing and e-auctions in IREPS. Covers registration, eligibility, participating in auctions, contract management, online payments, profile updates, compliance, and penalties.
    Main flows: Register; log in; browse/join auctions; submit bids; win/manage contracts; handle online payments; maintain profile/data.

    -----
    [Bidder_manual.pdf]
    Purpose: Bidder guidance for IREPS auctions—registration, account setup, bidding process, bid management, payments, messages, profile/data updates.
    Main flows: Register firm; login; update account; subscribe to depots/auctions; bid on lots; pay/refund via EMD/lien; manage awards and profile.

    -----
    [iMMS_HQ_Manual.pdf]
    Purpose: HQ user's manual for Integrated Materials Management System in Railways—covering demand registration, purchase proposals, tenders, inventory and item management, reporting, and order workflows.
    Main flows: Log in; register demand; create/authorize proposals; publish/modify tenders; manage purchase orders; update stock/item master; query/reporting.

    ----


    User Context Documents Provided Above. 
    Use ONLY this information for answering.

    User Question:
    {query}

    Answer:
    """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def chat(self, query: str) -> str:
        """Main chat function"""
        # Search for relevant documents
        search_results = self.search_documents(query)

        if not search_results['documents'][0]:
            return "I don't have any documents to search through. Please upload some documents first."

        # Combine context from search results
        context = ""
        sources = set()
        for i, doc in enumerate(search_results['documents'][0]):
            source = search_results['metadatas'][0][i]['source']
            sources.add(source)
            context += f"From {source}:\n{doc}\n\n"

        # Generate response
        response = self.generate_response(query, context)

        # Add sources
        if sources:
            response += f"\n\nSources: {', '.join(sources)}"

        return response
