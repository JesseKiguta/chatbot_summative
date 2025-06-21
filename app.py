# app.py
import streamlit as st
import json
import re
import random
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# ---------------- Load Data ----------------
@st.cache_data
def load_hotels():
    with open("hotels_data_large.json", "r") as f:
        return json.load(f)

hotels = load_hotels()

# List of cities used for extraction
cities_extended = list(set([hotel['city'] for hotel in hotels]))

# ---------------- Model Setup ----------------
model_path = "./trained_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
id2label = model.config.id2label

# ---------------- Intent Prediction ----------------
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
    return id2label[pred_id], round(confidence * 100, 1)

# ---------------- City Extraction ----------------
def extract_city(text: str) -> str:
    for city in cities_extended:
        if city.lower() in text.lower():
            return city
    return "Tokyo"  # Default fallback

# ---------------- Hotel Logic ----------------
def find_hotels_by_city(city: str, limit=3):
    return [hotel for hotel in hotels if hotel["city"].lower() == city.lower()][:limit]

def find_additional_hotels_in_city(city: str, exclude_hotels, limit=3):
    return [h for h in hotels if h["city"].lower() == city.lower() and h not in exclude_hotels][:limit]

# ---------------- Intent Response Generator ----------------
def generate_response(intent, user_input, chat_state):
    city = extract_city(user_input)

    if intent == "MakeBooking":
        matches = find_hotels_by_city(city)
        chat_state['last_hotels'] = matches
        chat_state['last_city'] = city
        if not matches:
            return f"Sorry, no hotels found in {city}.", []
        response = f"Here are some hotels in {city} you can book:\n"
        for h in matches:
            response += f"- {h['name']} (${h['price']}, {h['rating']}\u2605)\n"
        return response, matches

    elif intent == "RequestAmenitiesInfo":
        recent = chat_state.get('last_hotels', [])
        if not recent:
            return "I don’t have any hotels to show amenities for. Please search for hotels first.", []
        response = "Here are the amenities for the recent hotels you viewed:\n"
        for h in recent:
            amenities = h.get("amenities", [])
            response += f"- {h['name']}: {', '.join(amenities)}\n"
        return response, recent

    elif intent == "RequestOtherHotel":
        current_city = chat_state.get('last_city', extract_city(user_input))
        seen = chat_state.get('last_hotels', [])
        alt = find_additional_hotels_in_city(current_city, exclude_hotels=seen)
        if not alt:
            return f"No other hotels found in {current_city}.", []
        response = f"Here are some other hotels in {current_city}:\n"
        for h in alt:
            response += f"- {h['name']} (${h['price']}, {h['rating']}\u2605)\n"
        return response, alt

    elif intent == "DeclineBooking":
        return "No problem! Let us know when you're ready to book.", []

    elif intent == "ThankYou":
        return "You're welcome! Let me know if you need anything else.", []

    else:
        return "Sorry, I didn't understand that. Can you rephrase?", []

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Hotel Booking Chatbot")
st.markdown("""
    <style>
        .stChatMessage {
            background-color: #2e2e2e;
            color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .user {
            background-color: #1e3d59;
            color: #ffffff;
        }
        .bot {
            background-color: #592c2c;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Hotel Booking Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_state" not in st.session_state:
    st.session_state.chat_state = {}

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything about booking hotels:")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    intent, confidence = predict_intent(user_input)
    response, matched_hotels = generate_response(intent, user_input, st.session_state.chat_state)
    
    st.session_state.chat_history.append((user_input, response))

for user, bot in st.session_state.chat_history:
    st.markdown(f"<div class='stChatMessage user'><strong>You:</strong> {user}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='stChatMessage bot'><strong>Bot:</strong> {bot}</div>", unsafe_allow_html=True)

