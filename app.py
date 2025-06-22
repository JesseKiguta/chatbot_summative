import gradio as gr
import json
import re
import random
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# -------- Load Hotel Data --------
with open("hotels_data_large.json", "r") as f:
    hotels = json.load(f)

cities_extended = list(set([hotel["city"] for hotel in hotels]))

# -------- Load Model --------
model_path = "./trained_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
id2label = model.config.id2label

# -------- Intent Prediction --------
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

# -------- Hotel Logic --------
def extract_city(text: str) -> str | None:
    for city in cities_extended:
        if city.lower() in text.lower():
            return city
    return None

def find_hotels_by_city(city: str, limit=3):
    return [hotel for hotel in hotels if hotel["city"].lower() == city.lower()][:limit]

def recommend_other_hotels(current_city: str, limit=3):
    return random.sample(
        [h for h in hotels if h["city"].lower() != current_city.lower()],
        k=min(limit, len(hotels))
    )

def extract_amenities(hotel):
    return hotel.get("amenities", [])

# -------- Response Generator --------
chat_state = {"last_city": None, "last_hotels": []}

def generate_response(user_input):
    intent, confidence = predict_intent(user_input)
    city = extract_city(user_input)

    if intent == "Unknown":
        return "I'm not sure how to respond to that. Could you rephrase your question about hotel booking?"

    if intent == "MakeBooking":
        if not city:
            return "Could you specify the city you'd like to book a hotel in?"
        matched = find_hotels_by_city(city)
        if not matched:
            return f"Sorry, no hotels found in {city}."
        chat_state["last_city"] = city
        chat_state["last_hotels"] = matched
        response = f"Here are some hotels in {city} you can book:\n"
        for h in matched:
            response += f"- {h['name']} (${h['price']}, {h['rating']}★)\n"
        return response

    elif intent == "RequestAmenitiesInfo":
        if not chat_state["last_hotels"]:
            return "Please search for hotels first so I can list their amenities."
        response = "Here are the amenities for the recent hotels you viewed:\n"
        for h in chat_state["last_hotels"]:
            response += f"- {h['name']}: {', '.join(h['amenities'])}\n"
        return response

    elif intent == "RequestOtherHotel":
        if not city and chat_state["last_city"]:
            city = chat_state["last_city"]
        if not city:
            return "Could you specify the city you're interested in?"
        alternatives = recommend_other_hotels(city)
        response = f"Here are some other hotels outside of {city}:\n"
        for h in alternatives:
            response += f"- {h['name']} in {h['city']} (${h['price']})\n"
        return response

    elif intent == "DeclineBooking":
        return "No problem! Let us know when you're ready to book."

    elif intent == "ThankYou":
        return "You're welcome! Let me know if you need anything else."

    return "I'm not sure how to help with that. Can you try asking about hotels?"

# -------- Gradio Interface --------
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🧳 Hotel Booking Chatbot</h1>")

    chat_history = gr.Chatbot(label="Conversation")
    user_input = gr.Textbox(placeholder="Ask me anything about hotels...", show_label=False)
    send_button = gr.Button("Send")

    def handle_input(message, history):
        response = generate_response(message)
        history.append((message, response))
        return "", history

    send_button.click(fn=handle_input, inputs=[user_input, chat_history], outputs=[user_input, chat_history])
    user_input.submit(fn=handle_input, inputs=[user_input, chat_history], outputs=[user_input, chat_history])

# Run the Gradio app
if __name__ == "__main__":
    demo.launch(share=True)

