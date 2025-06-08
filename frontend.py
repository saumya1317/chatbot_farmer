import streamlit as st
from datetime import datetime, timedelta

def format_timestamp():
    """Format timestamp for messages"""
    now = datetime.now()
    if now.date() == datetime.today().date():
        return now.strftime("Today at %I:%M %p")
    elif now.date() == (datetime.today().date() - timedelta(days=1)):
        return now.strftime("Yesterday at %I:%M %p")
    else:
        return now.strftime("%d %b %Y at %I:%M %p")

def get_custom_css():
    """Return custom CSS for the app"""
    return """
    <style>
        /* Global font settings */
        * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Main container styling */
        .main-header {
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 1rem;
            background-color: #128C7E;
            border-radius: 10px;
        }
        
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2rem;
            font-weight: 600;
        }
        
        .main-header p {
            color: #E2F3F1;
            margin: 0.5rem 0 0;
            font-size: 1rem;
        }
        
        /* Sidebar styling */
        .sidebar-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #128C7E;
            margin-bottom: 0.5rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e0e0e0;
        }
        
        /* Input container styling */
        .input-container {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            margin-bottom: 0.5rem;
        }
        
        /* Message bubbles */
        .user-message {
            background-color: #DCF8C6;
            color: #303030;
            padding: 10px 15px;
            border-radius: 7.5px;
            margin: 4px 0;
            max-width: 70%;
            float: right;
            clear: both;
            box-shadow: 0 1px 1px rgba(0,0,0,0.1);
            position: relative;
            font-size: 0.95em;
            line-height: 1.5;
            word-wrap: break-word;
            white-space: pre-wrap;
            border: 1px solid #E2F3F1;
        }
        
        .user-message:after {
            content: '';
            position: absolute;
            right: -6px;
            top: 8px;
            width: 0;
            height: 0;
            border-top: 6px solid transparent;
            border-bottom: 6px solid transparent;
            border-left: 6px solid #DCF8C6;
        }
        
        .assistant-message {
            background-color: white;
            color: #303030;
            padding: 10px 15px;
            border-radius: 7.5px;
            margin: 4px 0;
            max-width: 70%;
            float: left;
            clear: both;
            box-shadow: 0 1px 1px rgba(0,0,0,0.1);
            position: relative;
            font-size: 0.95em;
            line-height: 1.5;
            word-wrap: break-word;
            white-space: pre-wrap;
            border: 1px solid #E2F3F1;
        }
        
        .assistant-message:after {
            content: '';
            position: absolute;
            left: -6px;
            top: 8px;
            width: 0;
            height: 0;
            border-top: 6px solid transparent;
            border-bottom: 6px solid transparent;
            border-right: 6px solid white;
        }
        
        /* Updated timestamp styling */
        .message-timestamp {
            font-size: 0.65em;
            color: #667781;
            margin-top: 4px;
            text-align: right;
            font-family: inherit;
            opacity: 0.8;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 2px;
        }
        
        .message-timestamp::before {
            content: '✓';
            font-size: 0.9em;
            color: #128C7E;
        }
        
        .user-message .message-timestamp::before {
            content: '✓✓';
            color: #128C7E;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #128C7E;
            color: white;
            border: none;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            font-weight: 500;
            font-size: 0.9em;
            transition: background-color 0.2s;
        }
        
        .stButton button:hover {
            background-color: #075E54;
        }
        
        /* File uploader styling */
        .stFileUploader {
            border: 1px dashed #128C7E;
            border-radius: 4px;
            padding: 0.5rem;
        }
        
        /* Text area styling */
        .stTextArea textarea {
            border-radius: 4px;
            border: 1px solid #128C7E;
            padding: 0.4rem;
            font-size: 0.95em;
            font-family: inherit;
        }
        
        /* Chat container styling */
        .chat-container {
            background-color: #E5DDD5;
            background-image: url('https://web.whatsapp.com/img/bg-chat-tile-light_a4be512e7195b6b733d9110b408f075d.png');
            background-repeat: repeat;
            padding: 0.8rem;
            border-radius: 8px;
            margin-top: 0.5rem;
            max-height: 500px;
            overflow-y: auto;
        }
        
        /* Column styling */
        .stColumn {
            padding: 0 0.5rem;
        }
        
        /* Markdown styling */
        .stMarkdown {
            font-size: 0.95em;
            font-family: inherit;
        }
        
        /* Success message styling */
        .stSuccess {
            font-size: 0.85em;
            padding: 0.3rem 0.5rem;
            background-color: #DCF8C6;
            color: #128C7E;
            border: 1px solid #128C7E;
            border-radius: 4px;
        }
        
        /* Warning message styling */
        .stWarning {
            font-size: 0.85em;
            padding: 0.3rem 0.5rem;
            background-color: #FFE5E5;
            color: #D32F2F;
            border: 1px solid #D32F2F;
            border-radius: 4px;
        }
        
        /* Subheader styling */
        .stSubheader {
            color: #128C7E;
            font-weight: 600;
            font-size: 1.1em;
        }
        
        /* Radio and selectbox styling */
        .stRadio > div {
            color: #128C7E;
        }
        
        .stSelectbox > div {
            color: #128C7E;
        }
        
        /* Custom scrollbar */
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: #128C7E;
            border-radius: 3px;
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: #075E54;
        }
    </style>
    """

def render_sidebar(theme, language, on_clear_chat):
    """Render the sidebar with settings"""
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Settings</div>', unsafe_allow_html=True)
        theme = st.radio("Theme", ["Light", "Dark"], index=0 if theme == 'light' else 1, label_visibility="collapsed")
        
        st.subheader("🌐 Language Settings")
        language = st.selectbox("Select language", ["English", "Hindi"], label_visibility="collapsed")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            on_clear_chat()
            st.rerun()
    
    return theme.lower(), language

def render_header():
    """Render the main header"""
    st.markdown('<div class="main-header"><h1>🌾 Farmer\'s Assistant</h1><p>Sustainable Practices & Disease Detection</p></div>', unsafe_allow_html=True)

def render_input_section(uploaded_file, uploaded_image, user_query, on_submit, language):
    """Render the input section with file uploads and query input"""
    left_col, middle_col, right_col = st.columns([1, 1, 1])
    
    with left_col:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.subheader("📄 Upload Documents")
        uploaded_file = st.file_uploader(
            "Upload PDF files about farming",
            type=["pdf"],
            label_visibility="collapsed",
            help="Upload farming-related PDF documents"
        )
        if uploaded_file is not None:
            st.success("PDF uploaded successfully!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with middle_col:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.subheader("🌿 Plant Analysis")
        uploaded_image = st.file_uploader(
            "Upload plant/leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Upload an image of a plant or leaf for analysis"
        )
        if uploaded_image is not None:
            st.success("Image uploaded successfully!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_col:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.subheader("💭 Ask Questions")
        user_query = st.text_area(
            "Enter your question about farming:",
            key="query_input",
            height=120,
            label_visibility="collapsed",
            help="Ask any question about farming practices"
        )
        
        if st.button("🌱 Get Advice", key="submit_button", use_container_width=True):
            if uploaded_image is None and not user_query.strip():
                st.warning("Please upload an image or enter your question.")
            else:
                on_submit(user_query, uploaded_image)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_file, uploaded_image, user_query

def render_chat_history(chat_history, language):
    """Render the chat history with messages"""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in chat_history:
        if message["role"] == "user":
            st.markdown(f'''
                <div class="user-message">
                    {message["content"]}
                    <div class="message-timestamp">{message["timestamp"]}</div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(f'''
                    <div class="assistant-message">
                        {message["content"]}
                        <div class="message-timestamp">{message["timestamp"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
            with col2:
                if st.button("📋", key=f"copy_{message['timestamp']}", help="Copy to clipboard", use_container_width=True):
                    escaped_content = message["content"].replace('"', '\\"').replace("'", "\\'")
                    st.write(f'<script>navigator.clipboard.writeText("{escaped_content}")</script>', unsafe_allow_html=True)
                    st.success("Copied!")
                if st.button("🔊", key=f"tts_{message['timestamp']}", help="Read message", use_container_width=True):
                    escaped_text = message["content"].replace('"', '\\"').replace("'", "\\'")
                    st.markdown(f"""
                    <script>
                        var msg = new SpeechSynthesisUtterance();
                        msg.text = "{escaped_text}";
                        msg.lang = "{'hi-IN' if language == 'Hindi' else 'en-US'}";
                        window.speechSynthesis.speak(msg);
                    </script>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    if 'language' not in st.session_state:
        st.session_state.language = 'English' 