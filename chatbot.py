import pickle
import random

# Load model, vectorizer, and responses
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("responses.pkl", "rb") as f:
    responses = pickle.load(f)

# Chat loop
print("Start chatting with the bot (type 'quit' to stop).")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    X_test = vectorizer.transform([user_input])
    predicted_tag = model.predict(X_test)[0]
    bot_response = random.choice(responses[predicted_tag])
    print("Bot:", bot_response)
