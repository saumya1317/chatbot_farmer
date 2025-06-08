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
    """Return custom CSS for the app with WhatsApp-style bubbles and improved layout"""
    return """
    <style>
    .chat-container {
        display: flex;
        flex-direction: column-reverse;
        gap: 0.5em;
        max-height: 60vh;
        overflow-y: auto;
        padding: 1em 0.5em;
        background: #ece5dd;
        border-radius: 12px;
        margin-bottom: 1em;
    }
    .whatsapp-bubble {
        display: flex;
        flex-direction: column;
        max-width: 70%;
        padding: 0.7em 1.1em 0.5em 1.1em;
        border-radius: 1.2em 1.2em 1.2em 0.3em;
        margin-bottom: 0.2em;
        font-size: 1em;
        position: relative;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        word-break: break-word;
    }
    .user-message.whatsapp-bubble {
        align-self: flex-end;
        background: #dcf8c6;
        border-bottom-right-radius: 0.3em;
        border-bottom-left-radius: 1.2em;
        margin-right: 0.2em;
    }
    .assistant-message.whatsapp-bubble {
        align-self: flex-start;
        background: #fff;
        border-bottom-left-radius: 0.3em;
        border-bottom-right-radius: 1.2em;
        margin-left: 0.2em;
    }
    .bubble-content {
        margin-bottom: 0.2em;
        white-space: pre-wrap;
    }
    .bubble-timestamp {
        font-size: 0.75em;
        color: #888;
        align-self: flex-end;
        margin-top: 0.1em;
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
        uploaded_file = st.file_uploader("Upload PDF files about farming", type=["pdf"], label_visibility="collapsed", help="Upload farming-related PDF documents")
        if uploaded_file is not None:
            st.success("PDF uploaded successfully!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with middle_col:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.subheader("🌿 Plant Analysis")
        uploaded_image = st.file_uploader("Upload plant/leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", help="Upload an image of a plant or leaf for analysis")
        if uploaded_image is not None:
            st.success("Image uploaded successfully!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_col:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.subheader("💭 Ask Questions")
        user_query = st.text_area("Enter your question about farming:", key="query_input", height=120, label_visibility="collapsed", help="Ask any question about farming practices")
        
        if st.button("🌱 Get Advice", key="submit_button", use_container_width=True):
            if uploaded_image is None and not user_query.strip():
                st.warning("Please upload an image or enter your question.")
            else:
                on_submit(user_query, uploaded_image)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_file, uploaded_image, user_query

def render_chat_history(chat_history, language):
    """Render the chat history in WhatsApp style, grouping each user question with its answer, newest pair first."""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Group messages as (user, assistant) pairs
    pairs = []
    i = 0
    while i < len(chat_history):
        if chat_history[i]["role"] == "user":
            user_msg = chat_history[i]
            # Check if next message is assistant
            if i + 1 < len(chat_history) and chat_history[i+1]["role"] == "assistant":
                assistant_msg = chat_history[i+1]
                pairs.append((user_msg, assistant_msg))
                i += 2
            else:
                pairs.append((user_msg, None))
                i += 1
        else:
            # If for some reason an assistant message is first, show it alone
            pairs.append((None, chat_history[i]))
            i += 1

    # Show newest pairs first
    for idx, (user_msg, assistant_msg) in enumerate(reversed(pairs)):
        if user_msg:
            st.markdown(f'''
                <div class="user-message whatsapp-bubble">
                    <span class="bubble-content">{user_msg["content"]}</span>
                    <span class="bubble-timestamp">{user_msg["timestamp"]}</span>
                </div>
            ''', unsafe_allow_html=True)
        if assistant_msg:
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(f'''
                    <div class="assistant-message whatsapp-bubble">
                        <span class="bubble-content">{assistant_msg["content"]}</span>
                        <span class="bubble-timestamp">{assistant_msg["timestamp"]}</span>
                    </div>
                ''', unsafe_allow_html=True)
            with col2:
                if st.button("📋", key=f"copy_{idx}_{assistant_msg['timestamp'] if assistant_msg else ''}", help="Copy to clipboard", use_container_width=True):
                    escaped_content = assistant_msg["content"].replace('"', '\\"').replace("'", "\\'")
                    st.write(f'<script>navigator.clipboard.writeText("{escaped_content}")</script>', unsafe_allow_html=True)
                    st.success("Copied!")
    st.markdown('</div>', unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
