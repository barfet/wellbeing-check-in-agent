import streamlit as st
import httpx
import logging
import os
from dotenv import load_dotenv
import asyncio
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv() # Load .env file if present
# Determine API base URL - default to local if not set
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_ENDPOINT = f"{API_BASE_URL}/api/v1/reflections/turns"

# Check if OPENAI_API_KEY is set (optional, for user awareness)
openai_api_key_present = bool(os.getenv("OPENAI_API_KEY"))

# --- Streamlit App ---
st.set_page_config(page_title="Reflection Agent", layout="centered")
st.title("Reflective Learning Agent")

if not openai_api_key_present:
    st.warning("Warning: OPENAI_API_KEY environment variable not found. The agent backend might not function correctly.")

st.info(f"Using API endpoint: {API_ENDPOINT}")

# --- Session State Initialization ---
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = None # Stores the 'next_state' dict from API
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores chat history for display [{role: "user"/"assistant", content: "..."}]
if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False

# --- Helper Function to Call API ---
async def call_reflection_api(topic: Optional[str] = None, 
                            user_input: Optional[str] = None, 
                            current_state: Optional[Dict[str, Any]] = None):
    payload = {
        "topic": topic,
        "user_input": user_input,
        "current_state": current_state
    }
    logger.info(f"Calling API endpoint: {API_ENDPOINT} with payload: {{topic: {topic}, user_input: {user_input[:50] if user_input else None}...}}")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout for LLM calls
            response = await client.post(API_ENDPOINT, json=payload)
        response.raise_for_status() # Raise exception for non-2xx status codes
        data = response.json()
        logger.info(f"API Response received: {{agent_response: {data.get('agent_response', '')[:50]}..., is_final: {data.get('is_final_turn')}}}")
        return data
    except httpx.RequestError as e:
        st.error(f"API Request Error: Could not connect to the backend at {API_ENDPOINT}. Please ensure the backend service is running. Details: {e}")
        logger.error(f"API Request Error: {e}")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API Error: Received status {e.response.status_code}. Response: {e.response.text}")
        logger.error(f"API HTTP Status Error: {e.response.status_code}, Response: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.exception(f"Unexpected error calling API: {e}")
        return None

# --- UI Rendering --- 

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Start conversation controls (only if not started)
if not st.session_state.conversation_started:
    st.markdown("### Start a New Reflection")
    topic_input = st.text_input("Enter a topic or experience to reflect on (optional):", key="topic_input")
    if st.button("Start Reflection", key="start_button"):
        st.session_state.conversation_started = True
        st.session_state.conversation_ended = False
        st.session_state.messages = [] # Clear previous messages
        st.session_state.conversation_state = None
        st.rerun() # Rerun to process the start

# Handle conversation flow if started
if st.session_state.conversation_started and not st.session_state.messages:
    # First turn after clicking start
    topic = st.session_state.get("topic_input", None)
    with st.spinner("Agent is thinking..."):
        api_response = asyncio.run(call_reflection_api(topic=topic))
        if api_response:
            st.session_state.conversation_state = api_response["next_state"]
            agent_response_content = api_response["agent_response"]
            st.session_state.messages.append({"role": "assistant", "content": agent_response_content})
            st.session_state.conversation_ended = api_response["is_final_turn"]
            # Rerun immediately to display the first agent message and input box
            st.rerun() 

# Chat input for subsequent turns (if conversation ongoing)
if st.session_state.conversation_started and not st.session_state.conversation_ended:
    prompt = st.chat_input("Your response:")
    if prompt:
        # Add user message to display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Call API with user input and current state
        with st.spinner("Agent is thinking..."):
            api_response = asyncio.run(
                call_reflection_api(
                    user_input=prompt, 
                    current_state=st.session_state.conversation_state
                )
            )
            if api_response:
                st.session_state.conversation_state = api_response["next_state"]
                agent_response_content = api_response["agent_response"]
                st.session_state.messages.append({"role": "assistant", "content": agent_response_content})
                st.session_state.conversation_ended = api_response["is_final_turn"]
                # Rerun to display the new agent message and potentially end the chat
                st.rerun()
            else:
                 # If API call fails, remove the user message we optimistically added
                 st.session_state.messages.pop()
                 st.error("Failed to get response from agent. Please check the backend logs.")

# Display end message or restart button
if st.session_state.conversation_ended:
    st.info("The reflection session has concluded.")
    if st.button("Start New Reflection", key="restart_button"):
        # Reset state for a new conversation
        st.session_state.conversation_state = None
        st.session_state.messages = []
        st.session_state.conversation_started = False
        st.session_state.conversation_ended = False
        # Clear topic input from previous session if needed (using key)
        # st.session_state.topic_input = "" # Or let it persist?
        st.rerun() 