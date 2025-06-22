![Screenshot 2025-06-21 111250](https://github.com/user-attachments/assets/1ce2545b-c67f-4e58-a05e-7a8b621a2e4d)# Hotel Booking Chatbot

A conversational AI chatbot for assisting users with hotel bookings. The bot predicts user intent (e.g., make a booking, request amenities info, etc.) and provides dynamic responses using a locally stored hotel dataset.

---

## Datasets

The first hotel dataset used for this project has 12520 rows was found on Hugging Face. This dataset was used to train the model. The dataset has the following columns:

* Domain. Hotels only, since the dataset is a hotel dataset.
* Context
* Response
* Act. These are the intents the context and response are classified under.
* Speaker. 0 for user and 1 for agent.

> **Link to dataset:** [SGD_Hotels Dataset](https://huggingface.co/datasets/vidhikatkoria/SGD_Hotels)

The second hotel dataset used for this project includes 2,000 hotels across 100 global cities. It is used by the trained model to give dynamic responses to the user. Each hotel entry includes:

* Hotel name
* City
* Price (USD)
* Star rating
* List of amenities

> **Link to dataset:** [hotels\_data\_large.json](https://huggingface.co/spaces/jessekdev/hotel-booking-chatbot/blob/main/hotels_data_large.json)


---

##  Project Structure

```
├── app.py                     # Gradio chatbot interface
├── hotels_data_large.json    # Local dataset used for responses
├── trained_model/            # BERT model directory
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.txt
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
```

---

##  Performance Metrics

The chatbot uses a fine-tuned BERT model for intent classification. Here are the model's performance metrics:

| Metric              | Value |
| ------------------- | ----- |
| Accuracy            | 0.76  |
| Precision           | 0.78  |
| Recall              | 0.76  |
| F1 Score            | 0.76  |
| Loss (CrossEntropy) | 0.3529|

Out-of-domain detection is implemented by rejecting predictions with confidence < 60%.

---

## Running the Chatbot

### Try the chatbot on Hugging Face Spaces

**[Deployed Version](https://huggingface.co/spaces/jessekdev/hotel-booking-chatbot)**

### Run locally

1. **Clone the repo**

```bash
git clone https://huggingface.co/spaces/jessekdev/hotel-booking-chatbot
cd hotel-booking-chatbot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
python app.py
```

---

## Example Conversations

| User Input                         | Bot Response                                                                                                                                                                                                                                                                                              |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| I'm looking for a hotel in Nairobi | Here are some hotels in Nairobi you can book: - Villanueva LLC Hotel (\$302, 4.3 stars) - Moore, Stewart and Day Hotel (\$269, 3.7 stars) - Miller-Smith Hotel (\$173, 5.0 stars)                                                                                                                                        |
| What utilities do they offer?      | Here are the amenities for the recent hotels you viewed: - Villanueva LLC Hotel: Airport Shuttle, Air Conditioning, Restaurant - Moore, Stewart and Day Hotel: Laundry Service, Parking, Breakfast, Gym, Non-smoking Rooms - Miller-Smith Hotel: Pet Friendly, Laundry Service, Spa, Gym, Airport Shuttle |
| Do you also do flights?            | I'm not sure how to respond to that. Could you rephrase your question about hotel booking?                                            


Screenshot of conversation (cont):
![Screenshot 2025-06-21 111250](https://github.com/user-attachments/assets/ab7e829e-c205-45d1-90c8-74ab4d967f47)

---

## Demo Video

**[Watch Demo Video](https://drive.google.com/file/d/1NytcbmW88cxc9Lc93OZz_KjpRqqkzwx1/view?usp=drive_link)**

---

## Contributors

* Jesse Kiguta [@jessekdev](https://huggingface.co/jessekdev)

---

## License

This project is licensed under the MIT License.

---

Feel free to fork, modify, and build upon this chatbot to suit your own domain needs!
