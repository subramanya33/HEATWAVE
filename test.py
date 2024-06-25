from datetime import datetime
from dateutil import parser
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the data from a CSV file
file_path = 'delhi15.csv'  # Update this path to where you saved the CSV
data = pd.read_csv(file_path)

# Select relevant columns and convert the datetime column
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
features = ['tempmax', 'tempmin', 'temp', 'humidity', 'windspeed', 'sealevelpressure']
date_input = input("Enter a date (YYYY-MM-DD): ")

# Parse the date input into a datetime object
try:
    user_date = parser.parse(date_input).date()
except ValueError:
    print("Invalid date format. Please enter date in YYYY-MM-DD format.")
    exit()

# Normalize the data
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Create sequences for LSTM input
def create_sequences(data, seq_length=5):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length][features].values
        label = data.iloc[i + seq_length]['tempmax']
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

seq_length = 5
X, y = create_sequences(data, seq_length)
y = (y > 40).astype(int)  # Label as 1 if tempmax > 40, else 0

# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the enhanced LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Predict future heatwaves
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Add predictions to the dataframe
data['predicted_heatwave'] = np.nan
data.loc[data.index[seq_length:seq_length+len(predicted_labels)], 'predicted_heatwave'] = predicted_labels

# Function to predict heatwaves based on user input dates
def predict_heatwave_for_dates(dates):
    # Ensure all requested dates are in the DataFrame
    valid_dates = [date for date in dates if date in data.index]
    if not valid_dates:
        print("No valid dates found in the data.")
        return pd.DataFrame()  # Return an empty DataFrame if no valid dates

    new_data = data.loc[valid_dates]
    new_data_scaled = scaler.transform(new_data[features])
    
    # Create sequences from the new data
    new_sequences = []
    for i in range(len(new_data_scaled) - seq_length):
        seq = new_data_scaled[i:i + seq_length]
        new_sequences.append(seq)
    
    if not new_sequences:
        print("Not enough data to create sequences.")
        return pd.DataFrame()  # Return an empty DataFrame if not enough data
    
    new_sequences = np.array(new_sequences)
    
    # Predict heatwaves
    new_predictions = model.predict(new_sequences)
    new_predicted_labels = (new_predictions > 0.5).astype(int).flatten()
    
    result = pd.DataFrame({
        'date': new_data.index[seq_length:],
        'tempmax': new_data['tempmax'][seq_length:],
        'temp': new_data['temp'][seq_length:],
        'humidity': new_data['humidity'][seq_length:],
        'predicted_heatwave': new_predicted_labels
    })
    
    result['heatwave'] = result['predicted_heatwave'].apply(lambda x: 'Yes' if x == 1 else 'No')
    
    return result

# Example usage with user input dates
user_dates = pd.date_range(start='2024-06-16', end='2024-06-25')
prediction_result = predict_heatwave_for_dates(user_dates)
print(prediction_result)

# Plot actual vs predicted heatwaves
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['tempmax'], label='Actual TempMax')
plt.plot(data.index, data['humidity'], label='Humidity')
valid_predictions = data['predicted_heatwave'].dropna()
plt.scatter(valid_predictions.index, valid_predictions * 40, label='Predicted Heatwave', color='red', marker='x')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual TempMax, Humidity and Predicted Heatwaves')
plt.legend()
plt.show()

# Save the model
model.save("lstmtest.h5")
