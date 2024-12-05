import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import random

# Define more example sentences for positive, negative, and neutral categories
positive_examples = [
    "I love this!", "This is amazing!", "I'm so happy!", "This is fantastic!", 
    "What a wonderful experience!", "I feel great!", "This is incredible!", 
    "Absolutely loved it!", "I had a great time!", "This made my day!",
    "I can't believe how good this is!", "So much fun!", "This is perfect!", 
    "What a pleasure!", "I'm really enjoying this!", "This is everything I wanted!",
    "It exceeded my expectations!", "I would highly recommend this!", "I am so pleased!"
]

negative_examples = [
    "I hate this!", "This is terrible!", "I'm so disappointed!", "This is awful!", 
    "What a horrible experience!", "I feel terrible!", "This is a nightmare!", 
    "Absolutely hated it!", "I had a bad time!", "This ruined my day!",
    "Worst experience ever!", "I regret this completely!", "This is the worst!", 
    "I don't like it at all!", "I feel miserable!", "This is completely useless!", 
    "I wouldn't recommend this to anyone!", "This is a disaster!", "Such a waste of time!"
]

neutral_examples = [
    "It's okay.", "Nothing special.", "It's fine.", "I don't have an opinion.", 
    "It's average.", "It is what it is.", "Just a normal experience.", 
    "I'm indifferent about it.", "It was neither good nor bad.", "It was alright.",
    "I guess it was fine.", "There's nothing remarkable about it.", "It's just okay.",
    "I don't feel strongly about it.", "It was decent.", "No big deal.", "It's just another thing.",
    "I don't mind it.", "It was a neutral experience.", "It was passable."
]

# Function to generate a much larger dataset
def generate_large_dataset(num_samples=10000):
    data = []
    labels = []

    for _ in range(num_samples // 3):  # Generate an equal number of positive, negative, and neutral samples
        data.append(random.choice(positive_examples))
        labels.append("positive")
        
        data.append(random.choice(negative_examples))
        labels.append("negative")
        
        data.append(random.choice(neutral_examples))
        labels.append("neutral")

    return data, labels

# Generate a dataset with 10,000 samples (3,333 for each class)
data, labels = generate_large_dataset(10000)

# Preview the generated dataset (first 10 samples)
for i in range(10):  # Print the first 10 samples
    print(f"Sentence: {data[i]}, Sentiment: {labels[i]}")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)
model = MultinomialNB()
model.fit(X, labels)

# Save model and vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("Model saved successfully as sentiment_model.pkl")
