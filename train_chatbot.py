import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load intents
with open("intents.json", "r") as f:
    data = json.load(f)

corpus = []
labels = []
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(pattern)
        labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Save model, vectorizer, and responses
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("responses.pkl", "wb") as f:
    pickle.dump(responses, f)

print("Training complete.")
