import json
import random
import nltk
import string

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents file
with open("intents.json") as file:
    intents = json.load(file)

# Preprocess user input
def clean_input(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
    return tokens

# Match input to intents
def classify_intent(user_input):
    tokens = clean_input(user_input)
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_tokens = clean_input(pattern)
            if set(pattern_tokens).intersection(tokens):
                return intent
    return None

# Main chat loop
def chat():
    print("🤖 ChatBot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("🤖 ChatBot: Bye! Have a nice day.")
            break
        intent = classify_intent(user_input)
        if intent:
            print("🤖 ChatBot:", random.choice(intent["responses"]))
        else:
            print("🤖 ChatBot: Sorry, I didn't understand that.")

if __name__ == "__main__":
    chat()
