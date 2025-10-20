import streamlit as st
import os
from rag_chatbot import RAGChatbot

# Page config
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize chatbot
@st.cache_resource
def load_chatbot():
    return RAGChatbot()

# Auto-index on startup
@st.cache_data
def auto_index_on_startup():
    chatbot = load_chatbot()
    return chatbot.auto_index_documents()

def main():
    st.title("📚 RAG Document Chatbot")
    # st.markdown("Place your PDF and TXT files in the `./documents` folder - they'll be automatically indexed!")

    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_google_gemini_api_key_here":
        st.error("⚠️ Please set your GOOGLE_API_KEY in the .env file")
        st.stop()

    # Initialize chatbot and auto-index documents
    chatbot = load_chatbot()

    # Auto-index documents on startup
    with st.spinner("Checking for new documents..."):
        auto_index_on_startup()

    # Sidebar for document management, sidebar closed by default
    with st.sidebar:
        st.header("📁 Document Management")

        # Auto-index button
        if st.button("🔄 Scan & Index Documents", help="Scan ./documents folder and index new/updated files"):
            status_placeholder = st.empty()

            def update_status(message):
                status_placeholder.text(message)

            with st.spinner("Scanning documents folder..."):
                new_files, updated_files = chatbot.auto_index_documents(progress_callback=update_status)

                if new_files:
                    st.success(f"✅ Indexed {len(new_files)} new files: {', '.join(new_files)}")
                if updated_files:
                    st.info(f"🔄 Updated {len(updated_files)} files: {', '.join(updated_files)}")
                if not new_files and not updated_files:
                    st.info("📄 All documents are up to date")

            status_placeholder.empty()

        # Show current documents
        st.subheader("📋 Current Documents")
        files_in_folder = chatbot.scan_documents_folder()
        if files_in_folder:
            for file_path in files_in_folder:
                filename = os.path.basename(file_path)
                if filename in chatbot.processed_files:
                    st.text(f"✅ {filename}")
                else:
                    st.text(f"⏳ {filename} (not indexed)")
        else:
            st.text("No files in ./documents folder")

        # Document stats
        try:
            collection_count = chatbot.collection.count()
            if collection_count > 0:
                st.info(f"📊 {collection_count} document chunks in database")
        except:
            st.info("📊 No documents processed yet")

        # Instructions
        st.markdown("---")
        st.markdown("**How to add documents:**")
        st.markdown("1. Place PDF/TXT files in `./documents/` folder")
        st.markdown("2. Click 'Scan & Index Documents'")
        st.markdown("3. Start asking questions!")

    # Main chat interface
    st.markdown("#### 💬 Chat with your documents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = chatbot.chat(prompt)
                # Find where 'Sources' is mentioned and make it smaller, bolder, and gray, italicized for better readability, Everything after 'Sources:' should be in smaller font size
                if "Sources:" in response:
                    response_parts = response.split("Sources:")
                    main_response = response_parts[0]
                    sources = "Sources:" + response_parts[1]
                    formatted_sources = f"<span style='font-size:0.9em; color:gray; font-style:italic;'><b>{sources}</b></span>"
                    response = main_response + "\n\n" + formatted_sources

                st.markdown(response, unsafe_allow_html=True)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
