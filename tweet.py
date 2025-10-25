import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import re

df = pd.read_csv("Tweets.csv")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

df['clean_text'] = df['text'].apply(clean_text)

vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['airline_sentiment'])
y = to_categorical(y)

X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(X, y, df['clean_text'], test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

optimizer = SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(y_test, axis=1)

predicted_sentiments = label_encoder.inverse_transform(predicted_labels)
actual_sentiments = label_encoder.inverse_transform(actual_labels)

for i in range(10):
    print(f"\nTweet: {text_test.iloc[i]}")
    print(f"Actual Sentiment: {actual_sentiments[i]}")
    print(f"Predicted Sentiment: {predicted_sentiments[i]}")