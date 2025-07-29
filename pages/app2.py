import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from PyPDF2 import PdfReader
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Streamlit app
st.title("Research Assistant with Gemini 1.5 Flash")
st.markdown("Ask research-related questions and get AI-powered responses!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a research assistant powered by Gemini 1.5 Flash. Provide accurate, detailed, and well-structured responses to research-related queries. Cite sources when possible and maintain a professional tone.")
    ]

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    max_tokens=1000
)

# Sidebar for file upload and processing options
with st.sidebar:
    st.header("Upload Document or Image")
    uploaded_file = st.file_uploader("Upload a document (PDF, TXT) or image (PNG, JPG, JPEG)", type=["pdf", "txt", "png", "jpg", "jpeg"])
    
    # Store uploaded content
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            st.session_state.uploaded_content = text
            st.success("PDF uploaded successfully!")
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            st.session_state.uploaded_content = text
            st.success("Text file uploaded successfully!")
        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.success("Image uploaded successfully!")
    
    # Processing option selection
    processing_option = st.selectbox(
        "How should I process your query?",
        ["Use LLM only", "Use uploaded content only", "Use both LLM and uploaded content"],
        key="processing_option"
    )

# Function to process query based on user selection
def process_query(prompt, processing_option, uploaded_content=None, uploaded_image=None):
    messages = st.session_state.messages.copy()
    
    if processing_option == "Use LLM only":
        messages.append(HumanMessage(content=prompt))
        response = llm.invoke(messages)
        return response.content
    elif processing_option == "Use uploaded content only":
        if uploaded_content:
            messages.append(HumanMessage(content=f"Using the provided document: {uploaded_content}\n\nAnswer the query: {prompt}"))
            response = llm.invoke(messages)
            return response.content
        elif uploaded_image:
            # Convert image to bytes for Gemini
            img_byte_arr = io.BytesIO()
            uploaded_image.save(img_byte_arr, format=uploaded_image.format)
            img_data = img_byte_arr.getvalue()
            messages.append(HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{uploaded_image.format.lower()};base64,{img_data}"}},
            ]))
            response = llm.invoke(messages)
            return response.content
        else:
            return "No uploaded content available. Please upload a document or image."
    else:  # Use both LLM and uploaded content
        combined_prompt = prompt
        if uploaded_content:
            combined_prompt = f"Using the provided document: {uploaded_content}\n\nAnswer the query: {prompt}"
        if uploaded_image:
            img_byte_arr = io.BytesIO()
            uploaded_image.save(img_byte_arr, format=uploaded_image.format)
            img_data = img_byte_arr.getvalue()
            messages.append(HumanMessage(content=[
                {"type": "text", "text": combined_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{uploaded_image.format.lower()};base64,{img_data}"}},
            ]))
        else:
            messages.append(HumanMessage(content=combined_prompt))
        response = llm.invoke(messages)
        return response.content

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content[0]["text"] if isinstance(message.content, list) else message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Handle user input
if prompt := st.chat_input("Enter your research query:"):
    # Add user message to history
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response based on processing option
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            uploaded_content = st.session_state.get("uploaded_content", None)
            uploaded_image = st.session_state.get("uploaded_image", None)
            response_text = process_query(prompt, processing_option, uploaded_content, uploaded_image)
            message_placeholder.markdown(response_text)
            
            # Add assistant response to history
            ai_message = AIMessage(content=response_text)
            st.session_state.messages.append(ai_message)
        except Exception as e:
            message_placeholder.error(f"Error: {str(e)}")
            st.error("An error occurred while processing your request. Please try again.")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = [
        SystemMessage(content="You are a research assistant powered by Gemini 1.5 Flash. Provide accurate, detailed, and well-structured responses to research-related queries. Cite sources when possible and maintain a professional tone.")
    ]
    if "uploaded_content" in st.session_state:
        del st.session_state.uploaded_content
    if "uploaded_image" in st.session_state:
        del st.session_state.uploaded_image
    st.experimental_rerun()