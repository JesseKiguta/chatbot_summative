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
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    if confidence < 0.6:
        return "Unknown", round(confidence * 100, 1)

    return id2label[pred_id], round(confidence * 100, 1)

# ---------------- City Extraction ----------------
def extract_city(text: str) -> str | None:
    for city in cities_extended:
        if city.lower() in text.lower():
            return city
    return None

# ---------------- Hotel Logic ----------------
def find_hotels_by_city(city: str, exclude: list = None, limit=3):
    if exclude is None:
        exclude = []
    return [hotel for hotel in hotels if hotel["city"].lower() == city.lower() and hotel['name'] not in exclude][:limit]

def recommend_other_hotels(current_city: str, current_hotels: list = None, limit=3):
    if current_hotels is None:
        current_hotels = []
    return random.sample([
        h for h in hotels 
        if h["city"].lower() == current_city.lower() and h['name'] not in current_hotels
    ], k=min(limit, len(hotels)))

# ---------------- Intent Response Generator ----------------
def generate_response(intent, user_input, chat_state):
    city = extract_city(user_input)

    # If city is required but missing, flag as unknown
    if intent in ["MakeBooking", "RequestOtherHotel", "RequestAmenitiesInfo"] and not city:
        return "I'm not sure what location you're referring to. Please mention a city so I can assist you.", []

    if intent == "MakeBooking":
        matches = find_hotels_by_city(city)
        chat_state['last_viewed_hotels'] = matches
        if not matches:
            return f"Sorry, no hotels found in {city}.", []
        response = f"Here are some hotels in {city} you can book:\n"
        for h in matches:
            response += f"- {h['name']} ($${h['price']}, {h['rating']}★)\n"
        return response.strip(), matches

    elif intent == "RequestAmenitiesInfo":
        last_hotels = chat_state.get('last_viewed_hotels', [])
        if not last_hotels:
            return "Please view some hotels first before requesting amenities.", []
        response = "Here are the amenities for the recent hotels you viewed:\n"
        for hotel in last_hotels:
            amenities = hotel.get("amenities", [])
            response += f"- {hotel['name']}: {', '.join(amenities)}\n"
        return response.strip(), last_hotels

    elif intent == "RequestOtherHotel":
        recent = chat_state.get('last_viewed_hotels', [])
        recent_names = [h['name'] for h in recent]
        alt = recommend_other_hotels(city, recent_names)
        if not alt:
            return f"No other hotels found in {city}.", []
        chat_state['last_viewed_hotels'] = alt
        response = f"Here are some other hotels in {city}:\n"
        for h in alt:
            response += f"- {h['name']} ($${h['price']}, {h['rating']}★)\n"
        return response.strip(), alt

    elif intent == "DeclineBooking":
        return "No problem! Let us know when you're ready to book.", []

    elif intent == "ThankYou":
        return "You're welcome! Let me know if you need anything else.", []

    elif intent == "Unknown":
        return "I'm not sure I understand. Could you rephrase or ask something related to hotels?", []

    else:
        return "Sorry, I didn't understand that. Can you rephrase?", []

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Hotel Chatbot", page_icon="🏨")

st.markdown("""
    <style>
        .stChatMessage { background-color: #333; padding: 10px; border-radius: 10px; margin-bottom: 10px; color: white; }
        .user { background-color: #1f5c4d; color: white; }
        .bot { background-color: #663c43; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("Hotel Booking Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store last viewed hotels for amenities lookup
if "chat_state" not in st.session_state:
    st.session_state.chat_state = {}

user_input = st.text_input("Ask me anything about booking hotels:", key="user_input")
send_clicked = st.button("Send")

if send_clicked and user_input:
    intent, confidence = predict_intent(user_input)
    response, matched_hotels = generate_response(intent, user_input, st.session_state.chat_state)

    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("bot", response))

# Display chat history
for sender, msg in st.session_state.messages:
    role_class = "user" if sender == "user" else "bot"
    st.markdown(f'<div class="stChatMessage {role_class}">{msg}</div>', unsafe_allow_html=True)

