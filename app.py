import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()
st.set_page_config(page_title="Research Assistant", page_icon=":robot_face:", layout="wide")

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.text_input("Enter your Google API Key", type="password")
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

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
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
    
    # Get response from Gemini
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = llm.invoke(st.session_state.messages)
            message_placeholder.markdown(response.content)
            
            # Add assistant response to history
            ai_message = AIMessage(content=response.content)
            st.session_state.messages.append(ai_message)
        except Exception as e:
            message_placeholder.error(f"Error: {str(e)}")
            st.error("An error occurred while processing your request. Please try again.")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = [
        SystemMessage(content="You are a research assistant powered by Gemini 1.5 Flash. Provide accurate, detailed, and well-structured responses to research-related queries. Cite sources when possible and maintain a professional tone.")
    ]
    st.experimental_rerun()
