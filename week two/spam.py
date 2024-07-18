import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

path = os.path.dirname(os.path.abspath(__file__))
spam = path + '/' + 'emails.csv'

data = pd.read_csv(spam, encoding='latin-1')

# Keep only necessary columns and rename them
data = data[['text', 'spam']]
data.columns = ['text', 'label']

# Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Ensure X_train and X_test are strings
X_train = X_train.astype(str)
X_test = X_test.astype(str)

# Tokenize the text
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding='post', truncating='post')

# Build the model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),  # Increased embedding dimension
    LSTM(128, return_sequences=True),  # Increased LSTM units
    Dropout(0.  ),
    LSTM(64),  # Reduced LSTM units for faster training
    Dropout(0.5),
    Dense(64, activation='relu'),  # Increased dense layer units
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train_padded, y_train, epochs=20, validation_data=(X_test_padded, y_test), batch_size=64, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Function to predict spam or ham
def predict_spam(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    return 'Spam' if prediction[0][0] > 0.5 else 'Not spam'

# Test with new data
new_text = "Congratulations! You just won ten free tickets to a baseball game! Click the link below!"
print(predict_spam(new_text))







