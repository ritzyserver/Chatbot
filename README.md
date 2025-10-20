# RAG Document Chatbot

A fast, accurate RAG-based chatbot that answers questions strictly based on your uploaded documents.

## Features

- **Document-only responses**: No hallucination, only answers from your docs
- **Multiple file support**: PDF and TXT files
- **Fast retrieval**: ChromaDB vector database for quick searches
- **Source attribution**: Shows which documents answers come from
- **Clean UI**: Simple Streamlit interface

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google Gemini API**:
   - Get your API key from [Google AI Studio](https://aistudio.google.com/prompts/new_chat)
   - Update `.env` file:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Add Documents**: Place your PDF or TXT files in the `./documents/` folder
2. **Auto-Index**: Documents are automatically scanned on startup, or click "Scan & Index Documents"
3. **Chat**: Ask questions about your documents in the chat interface

The system automatically detects new or updated files and re-indexes them as needed.

## Tech Stack

- **AI Model**: Google Gemini Pro
- **Vector DB**: ChromaDB (persistent storage)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **UI**: Streamlit
- **Document Processing**: PyPDF2

## Key Benefits

- ✅ Zero hallucination - only document-based answers
- ✅ Fast response times
- ✅ Persistent document storage
- ✅ Source citations
- ✅ Easy to use interface