import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import GooglePalm
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
from PIL import Image
import io
import base64
from frontend import (
    format_timestamp, get_custom_css, render_sidebar, render_header,
    render_input_section, render_chat_history, initialize_session_state
)

# Load environment variables
load_dotenv()

# Configure Google API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Language translations
TRANSLATIONS = {
    "English": {
        "upload_success": "PDF uploaded successfully!",
        "image_success": "Image uploaded successfully!",
        "enter_question": "Enter your question about farming:",
        "get_advice": "Get Advice",
        "upload_warning": "Please upload an image or enter your question.",
        "copied": "Copied!",
        "clear_chat": "Clear Chat History",
        "settings": "Settings",
        "language_settings": "Language Settings",
        "select_language": "Select language",
        "upload_documents": "Upload Documents",
        "upload_pdf": "Upload PDF files about farming",
        "plant_analysis": "Plant Analysis",
        "upload_image": "Upload plant/leaf image",
        "ask_questions": "Ask Questions",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark"
    },
    "Hindi": {
        "upload_success": "PDF सफलतापूर्वक अपलोड हो गया!",
        "image_success": "छवि सफलतापूर्वक अपलोड हो गई!",
        "enter_question": "कृषि के बारे में अपना प्रश्न दर्ज करें:",
        "get_advice": "सलाह प्राप्त करें",
        "upload_warning": "कृपया एक छवि अपलोड करें या अपना प्रश्न दर्ज करें।",
        "copied": "कॉपी किया गया!",
        "clear_chat": "चैट इतिहास साफ़ करें",
        "settings": "सेटिंग्स",
        "language_settings": "भाषा सेटिंग्स",
        "select_language": "भाषा चुनें",
        "upload_documents": "दस्तावेज़ अपलोड करें",
        "upload_pdf": "कृषि के बारे में PDF फ़ाइलें अपलोड करें",
        "plant_analysis": "पौधा विश्लेषण",
        "upload_image": "पौधे/पत्ती की छवि अपलोड करें",
        "ask_questions": "प्रश्न पूछें",
        "theme": "थीम",
        "light": "लाइट",
        "dark": "डार्क"
    }
}

def get_translated_text(key, language="English"):
    """Get translated text for the given key and language"""
    return TRANSLATIONS.get(language, TRANSLATIONS["English"]).get(key, key)

def check_api_key():
    """Check if API key is valid and has remaining quota"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Test")
        return True, None
    except Exception as e:
        if "quota" in str(e).lower():
            return False, "API quota exceeded. Please try again later."
        elif "invalid" in str(e).lower():
            return False, "Invalid API key. Please check your API key in the .env file."
        return False, f"API Error: {str(e)}"

def handle_api_error(error):
    """Handle API errors and return user-friendly messages"""
    error_str = str(error).lower()
    if "quota" in error_str:
        return "API quota exceeded. Please try again later."
    elif "invalid" in error_str:
        return "Invalid API key. Please check your API key in the .env file."
    elif "safety" in error_str:
        return "The content was blocked for safety reasons. Please try rephrasing your question."
    return f"An error occurred: {str(error)}"

def process_pdf(uploaded_file):
    """Process uploaded PDF file"""
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load and process the PDF
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(pages)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
        
        # Clean up temporary file
        os.remove("temp.pdf")
        
        return vector_store
    return None

def process_image(uploaded_image):
    """Process uploaded image for plant disease detection"""
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Convert to base64 for the model
        img_base64 = base64.b64encode(img_byte_arr).decode()
        
        return img_base64
    return None

def get_plant_analysis(image_base64, language="English"):
    """Get plant disease analysis using Gemini Pro Vision"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        image_parts = [{"mime_type": "image/jpeg", "data": image_base64}]
        
        prompt = "Analyze this plant/leaf image and identify any diseases or issues. Provide detailed information about the condition and suggest sustainable treatment methods. Format the response in a clear, structured way. Always focus on sustainable farming practices. Begin your answer with: 'Hello Kisaan 👨‍🌾'"
        if language == "Hindi":
            prompt = "इस पौधे/पत्ती की छवि का विश्लेषण करें और किसी भी बीमारी या समस्या की पहचान करें। स्थिति के बारे में विस्तृत जानकारी दें और स्थायी उपचार विधियों का सुझाव दें। प्रतिक्रिया को स्पष्ट, संरचित तरीके से प्रारूपित करें। हमेशा टिकाऊ कृषि पद्धतियों पर ध्यान केंद्रित करें। अपने उत्तर की शुरुआत करें: 'नमस्ते किसान 👨‍🌾'"
        
        response = model.generate_content([prompt, *image_parts])
        return response.text
    except Exception as e:
        return handle_api_error(e)

def get_farming_advice(query, vector_store, language="English"):
    """Get farming advice using the conversational chain, always focusing on sustainable farming and greeting the farmer."""
    try:
        # Initialize the language model
        llm = GooglePalm(google_api_key=GOOGLE_API_KEY, temperature=0.7)
        
        # Create memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True
        )
        
        # Add system prompt for sustainability and greeting
        system_prompt = "You are an expert in sustainable farming. Always provide answers that are relevant to sustainable agriculture. Begin every answer with: 'Hello Kisaan 👨‍🌾' and then answer the question."
        if language == "Hindi":
            system_prompt = "आप स्थायी कृषि के विशेषज्ञ हैं। हमेशा ऐसे उत्तर दें जो स्थायी कृषि से संबंधित हों। हर उत्तर की शुरुआत करें: 'नमस्ते किसान 👨‍🌾' और फिर प्रश्न का उत्तर दें।"
        
        # Get response
        response = chain({"question": f"{system_prompt}\n{query}"})
        return response['answer']
    except Exception as e:
        return handle_api_error(e)

def handle_submit(user_query, uploaded_image, vector_store, language):
    """Handle form submission and generate response"""
    if uploaded_image is not None:
        # Process image for plant disease detection
        image_base64 = process_image(uploaded_image)
        if image_base64:
            # Add user message to chat
            st.session_state.chat_history.append({
                "role": "user",
                "content": get_translated_text("upload_image", language),
                "timestamp": format_timestamp()
            })
            
            # Get and add analysis
            analysis = get_plant_analysis(image_base64, language)
            # Prepend greeting if not present
            if language == "Hindi":
                if not analysis.strip().startswith("नमस्ते किसान"):
                    analysis = f"नमस्ते किसान 👨‍🌾\n" + analysis
            else:
                if not analysis.strip().startswith("Hello Kisaan"):
                    analysis = f"Hello Kisaan 👨‍🌾\n" + analysis
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": analysis,
                "timestamp": format_timestamp()
            })
    
    if user_query.strip():
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": format_timestamp()
        })
        
        # Get and add response
        if vector_store:
            response = get_farming_advice(user_query, vector_store, language)
        else:
            # If no PDF is uploaded, use Gemini Pro directly
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = "You are an expert in sustainable farming. Always provide answers that are relevant to sustainable agriculture. Begin every answer with: 'Hello Kisaan 👨‍🌾' and then answer the question. "
                if language == "Hindi":
                    prompt = "आप स्थायी कृषि के विशेषज्ञ हैं। हमेशा ऐसे उत्तर दें जो स्थायी कृषि से संबंधित हों। हर उत्तर की शुरुआत करें: 'नमस्ते किसान 👨‍🌾' और फिर प्रश्न का उत्तर दें। "
                response = model.generate_content(f"{prompt}\n{user_query}").text
            except Exception as e:
                response = handle_api_error(e)
        # Prepend greeting if not present
        if language == "Hindi":
            if not response.strip().startswith("नमस्ते किसान"):
                response = f"नमस्ते किसान 👨‍🌾\n" + response
        else:
            if not response.strip().startswith("Hello Kisaan"):
                response = f"Hello Kisaan 👨‍🌾\n" + response
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": format_timestamp()
        })
    
    st.rerun()

def main():
    # Initialize session state
    initialize_session_state()
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Render header
    render_header()
    
    # Initialize variables
    uploaded_file = None
    uploaded_image = None
    user_query = ""
    vector_store = None
    
    # Check API key
    api_valid, api_error = check_api_key()
    if not api_valid:
        st.error(api_error)
        return
    
    # Render sidebar and get settings
    theme, language = render_sidebar(
        st.session_state.theme,
        st.session_state.get('language', 'English'),
        lambda: setattr(st.session_state, 'chat_history', [])
    )
    
    # Update session state
    st.session_state.theme = theme
    st.session_state.language = language
    
    # Get translated text for UI elements
    translations = TRANSLATIONS[language]
    
    # Render input section with translated text
    uploaded_file, uploaded_image, user_query = render_input_section(
        uploaded_file,
        uploaded_image,
        user_query,
        lambda q, img: handle_submit(q, img, vector_store, language),
        language
    )
    
    # Process PDF if uploaded
    if uploaded_file is not None:
        vector_store = process_pdf(uploaded_file)
    
    # Render chat history
    render_chat_history(st.session_state.chat_history, language)

if __name__ == "__main__":
    main() 