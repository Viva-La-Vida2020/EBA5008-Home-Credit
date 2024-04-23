import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split

def train_rnn(X_train, y_train):
    X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    model = Sequential([
        SimpleRNN(50, input_shape=(X_train_part.shape[1], X_train_part.shape[2]), return_sequences=True),
        SimpleRNN(50),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_part, y_train_part, epochs=10, batch_size=32, validation_data=(X_val_part, y_val_part))
    model.save('rnn_model.h5')