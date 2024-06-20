import streamlit as st
import pandas as pd
import vertexai  # Vertex AI library for interacting with AI models
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

# Initialize Vertex AI environment
vertexai.init(project="numeric-ion-425514-k6", location="us-central1")

# Initialize Generative Model
model = GenerativeModel("gemini-1.0-pro-001")
chat = model.start_chat()

# Generation configuration and safety settings
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# add info to csv
try:
    current_info_df = pd.read_csv('current_info.csv')
    if 'question' not in current_info_df.columns or 'answer' not in current_info_df.columns:
        st.error("Not enough data to answer")
        st.stop()
except Exception as e:
    st.error(f"Error loading CSV file: {str(e)}")
    st.stop()

# Streamlit interface
st.title("Chatbot Interface")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def get_answer_from_csv(user_question):
    # Check if the question exists in the CSV file
    question_lower = user_question.strip().lower()
    for index, row in current_info_df.iterrows():
        # if the user question and question in csv match up
        if 'question' in row and row['question'].strip().lower() == question_lower:
            return row['answer']
    return None

def get_chat_response(user_input):

    answer_from_csv = get_answer_from_csv(user_input)
    if answer_from_csv:
        return answer_from_csv

    # Construct the conversation context
    context_list = []
    for msg in st.session_state['chat_history']:
        context_list.append(f"You: {msg['user_input']}")
        context_list.append(f"Bot: {msg['chatbot_response']}")
    context = "\n".join(context_list)

    # Use the suggested format
    full_input = f"""You are AI Chatbot designed to answer the user question from the context provided
User Question: {user_input}

Context: {context}
"""

    # Send the context and the latest user input to the model
    response = chat.send_message([full_input], generation_config=generation_config, safety_settings=safety_settings)
    return response.candidates[0].content.parts[0].text

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response = get_chat_response(user_input)
        st.session_state['chat_history'].append({"user_input": user_input, "chatbot_response": response})

if st.session_state['chat_history']:
    for message in st.session_state['chat_history']:
        st.write(f"You: {message['user_input']}")
        st.write(f"Bot: {message['chatbot_response']}")
