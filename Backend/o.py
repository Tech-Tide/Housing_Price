import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Generate synthetic dataset
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, 100)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define a function to create the neural network model
def create_model(use_dropout=False):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(1,)))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    return model

# Train a neural network without dropout
model_no_dropout = create_model()
model_no_dropout.compile(optimizer='adam', loss='mean_squared_error')
history_no_dropout = model_no_dropout.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), verbose=0)

# Train a neural network with dropout
model_with_dropout = create_model(use_dropout=True)
model_with_dropout.compile(optimizer='adam', loss='mean_squared_error')
history_with_dropout = model_with_dropout.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), verbose=0)

# Plot the training and validation loss for both models
plt.figure(figsize=(12, 6))
plt.plot(history_no_dropout.history['loss'], label='Training Loss (No Dropout)')
plt.plot(history_no_dropout.history['val_loss'], label='Validation Loss (No Dropout)')
plt.plot(history_with_dropout.history['loss'], label='Training Loss (With Dropout)')
plt.plot(history_with_dropout.history['val_loss'], label='Validation Loss (With Dropout)')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
