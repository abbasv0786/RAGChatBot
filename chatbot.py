import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Together
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import nest_asyncio
import os
import time
import re
from urllib.parse import urlparse, parse_qs

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# API keys setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean Light Theme CSS
st.markdown("""
<style>
    /* Light theme colors */
    :root {
        --primary: #1f2937;
        --primary-light: #374151;
        --secondary: #6b7280;
        --accent: #3b82f6;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --background: #ffffff;
        --surface: #f9fafb;
        --text-primary: #111827;
        --text-secondary: #6b7280;
        --border: #d1d5db;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    /* Global styles */
    .main {
        background: var(--background);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Navigation */
    .nav-container {
        background: white;
        border-bottom: 1px solid var(--border);
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .nav-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .nav-brand {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    /* Header */
    .header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        padding: 3rem 0;
        text-align: center;
        margin-bottom: 2rem;
        border-radius: 0 0 20px 20px;
    }
    
    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Container */
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px var(--shadow);
        border: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    
    .card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    /* Status */
    .status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .status.ready {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status.processing {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .status-indicator.ready {
        background: var(--success);
    }
    
    .status-indicator.processing {
        background: var(--warning);
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Chat messages */
    .message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    
    .message.user {
        background: var(--accent);
        color: white;
        margin-left: auto;
    }
    
    .message.assistant {
        background: var(--surface);
        color: var(--text-primary);
        border: 1px solid var(--border);
        margin-right: auto;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2rem;
        }
        
        .nav-content {
            padding: 0 1rem;
        }
        
        .container {
            padding: 0 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = False
if "source_type" not in st.session_state:
    st.session_state.source_type = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Upload"

def get_conversation_chain(vectorstore):
    try:
        llm = ChatGoogleGenerativeAI(
            temperature=0.7,
            model='gemini-1.5-flash',
            convert_system_message_to_human=True
        )
        
        template = """You are a helpful AI assistant that helps users understand content from various sources.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always maintain a professional and helpful tone.
        
        Context: {context}
        
        Question: {question}
        Helpful Answer:"""

        prompt = PromptTemplate(
            input_variables=['context', 'question'],
            template=template
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt},
            return_source_documents=True
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    try:
        if 'youtube.com' in url:
            query = urlparse(url).query
            return parse_qs(query)['v'][0]
        elif 'youtu.be' in url:
            return url.split('/')[-1]
    except:
        return None
    return None

def get_youtube_transcript(url):
    """Get YouTube transcript using youtube_transcript_api directly"""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return None
            
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        st.error(f"Error getting YouTube transcript: {str(e)}")
        return None

def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text() or ""
                text += extracted_text.encode('utf-8', errors='ignore').decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def get_web_content(url):
    """Get web content using WebBaseLoader"""
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return ' '.join([doc.page_content for doc in data])
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")
        return None
    
def get_document_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return None

def create_vectorstore(chunks):
    """Create FAISS vectorstore for all content types"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def process_content(content, source_type, source_identifier):
    """Process content and store source information"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(f"Processing {source_type}...")
        progress_bar.progress(20)
        
        if not content:
            return False
            
        status_text.text("Splitting content into chunks...")
        progress_bar.progress(40)
        chunks = get_document_chunks(content)
        if not chunks:
            return False
            
        status_text.text("Creating embeddings and vectorstore...")
        progress_bar.progress(60)
        
        vectorstore = create_vectorstore(chunks)
        if not vectorstore:
            return False
            
        st.session_state.vectorstore = vectorstore
        
        status_text.text("Setting up conversation chain...")
        progress_bar.progress(90)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        st.session_state.processComplete = True
        st.session_state.source_type = source_type
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return False

def render_navigation():
    """Render the navigation bar"""
    st.markdown("""
    <div class="nav-container">
        <div class="nav-content">
            <div class="nav-brand">🧠 AI Research Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 Upload Content", use_container_width=True):
            st.session_state.current_page = "Upload"
            st.rerun()
    
    with col2:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.current_page = "Chat"
            st.rerun()

def render_upload():
    """Render the upload content page"""
    st.markdown("""
    <div class="header">
        <div class="container">
            <h1>📁 Upload Content</h1>
            <p>Upload documents, websites, or videos to analyze with AI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicator
    if st.session_state.processComplete:
        st.markdown("""
        <div class="status ready">
            <div class="status-indicator ready"></div>
            Ready to Chat
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status processing">
            <div class="status-indicator processing"></div>
            Processing Content
        </div>
        """, unsafe_allow_html=True)
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📁 Source Selection")
        
        source_type = st.selectbox(
            "Content Source",
            ["PDF Documents", "Website URL", "YouTube Video"],
            help="Select the type of content you want to analyze"
        )

        llm_model = st.selectbox(
            "🤖 AI Model",
            ["Gemini Pro", "Mixtral 8x7B", "LLaMA 2 70B", "Qwen 32B"],
            help="Choose the AI model for processing"
        )
        
        if source_type == "PDF Documents":
            uploaded_files = st.file_uploader(
                "Upload PDFs",
                type="pdf",
                accept_multiple_files=True,
                help="Upload one or more PDF files to analyze"
            )
            
            if uploaded_files:
                st.write("**Uploaded Files:**")
                for file in uploaded_files:
                    st.write(f"• {file.name}")
                
                process_button = st.button("Process PDFs", disabled=not uploaded_files)
                if process_button:
                    with st.spinner("Processing PDFs..."):
                        content = get_pdf_text(uploaded_files)
                        success = process_content(content, "PDF", ', '.join([f.name for f in uploaded_files]))
                        if success:
                            st.success("✅ Documents processed successfully!")
                            
        elif source_type == "Website URL":
            url = st.text_input("Website URL", key="web_url", placeholder="https://example.com")
            process_button = st.button("Process Website", disabled=not url)
            if process_button and url:
                with st.spinner("Processing website..."):
                    content = get_web_content(url)
                    success = process_content(content, "Website", url)
                    if success:
                        st.success("✅ Website processed successfully!")
                        
        elif source_type == "YouTube Video":
            url = st.text_input("YouTube URL", key="youtube_url", placeholder="https://youtube.com/watch?v=...")
            process_button = st.button("Process Video", disabled=not url)
            if process_button and url:
                with st.spinner("Processing video..."):
                    content = get_youtube_transcript(url)
                    success = process_content(content, "YouTube", url)
                    if success:
                        st.success("✅ Video processed successfully!")

    with col2:
        if not st.session_state.processComplete:
            st.markdown("""
            <div class="card">
                <div class="card-header">🎯 Ready to Upload?</div>
                <p>Select a source from the sidebar and upload your content to begin an intelligent conversation with your documents, websites, or videos.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">✅ Content Processed</div>
                <p>Your {st.session_state.source_type} content has been successfully processed and is ready for chat!</p>
                <p style="color: var(--success); font-weight: 600;">Go to the Chat page to start asking questions about your content.</p>
            </div>
            """, unsafe_allow_html=True)

def render_chat():
    """Render the chat page"""
    st.markdown("""
    <div class="header">
        <div class="container">
            <h1>💬 Chat</h1>
            <p>Ask questions about your uploaded content</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.processComplete:
        st.markdown("""
        <div class="card">
            <div class="card-header">📝 No Content Available</div>
            <p>Please upload some content first before you can start chatting. Go to the Upload Content page to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card">
            <div class="card-header">💬 Chat with {st.session_state.source_type}</div>
            <p>Ask questions about your content and get intelligent responses</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        for role, message in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"""
                <div class="message user">
                    <strong>You:</strong><br>
                    {message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message assistant">
                    <strong>AI Assistant:</strong><br>
                    {message}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_question = st.chat_input("Ask a question about your content...", key="chat_input")
        
        if user_question:
            # Display user message
            st.markdown(f"""
            <div class="message user">
                <strong>You:</strong><br>
                {user_question}
            </div>
            """, unsafe_allow_html=True)
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.conversation({
                        "question": user_question
                    })
                    bot_response = response["answer"]
                    
                    # Display AI response
                    st.markdown(f"""
                    <div class="message assistant">
                        <strong>AI Assistant:</strong><br>
                        {bot_response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.session_state.chat_history.append(("You", user_question))
                    st.session_state.chat_history.append(("Bot", bot_response))
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

def main():
    # Render navigation
    render_navigation()
    
    # Render current page based on navigation
    if st.session_state.current_page == "Upload":
        render_upload()
    elif st.session_state.current_page == "Chat":
        render_chat()

if __name__ == "__main__":
    main()