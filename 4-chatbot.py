import streamlit as st
import requests

st.title("Customer Churn Prediction Chatbot HCL Hackathon")

def get_prediction(description):
    """
    Send a POST request to the FastAPI endpoint with the customer description to get the churn prediction.

    Args:
        description (str): The customer description input by the user.

    Returns:
        dict: The response from the FastAPI endpoint, either as a JSON object or a string.
    """
    response = requests.post("http://127.0.0.1:8000/predict-from-text", json={"customer_description": description})
    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            return {"detail": response.text}
    else:
        return {"error": "Error fetching prediction"}

# Initialize session state for storing chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Capture user input
user_input = st.chat_input("Enter customer description:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get the prediction from the FastAPI endpoint
    response = get_prediction(user_input)
    if isinstance(response, dict):
        response_content = response.get("detail", "Sorry, there was an error!")
    else:
        response_content = response
    
    # Add the bot's response to the session state
    st.session_state.messages.append({"role": "bot", "content": response_content})

# Display the chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])