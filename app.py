import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load dataset
df = pd.read_csv("mental_health_dataset.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])
df["Diagnosis"] = label_encoder.fit_transform(df["Diagnosis"])

# Split features and target
X = df.drop(columns=["Patient_ID", "Diagnosis"])
y = df["Diagnosis"]

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Multi-class classification
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict a sample input
# Encode categorical value for 'Gender'
# Ensure label_encoder has seen the original classes
if "Female" not in label_encoder.classes_:
    label_encoder.classes_ = np.append(label_encoder.classes_, "Female")  # Add missing class
if "Male" not in label_encoder.classes_:
    label_encoder.classes_ = np.append(label_encoder.classes_, "Male")  # Add missing class

female_encoded = label_encoder.transform(["Male"])[0]  # Get numeric encoding for 'Female'

# Define sample input with numerical encoding
sample_input = np.array([[34, female_encoded, 7, 5, 4, 5, 75, 0]])
# Convert sample input to DataFrame with matching column names
sample_input_df = pd.DataFrame(sample_input, columns=["Age", "Gender", "Sleep_Hours", "Stress_Level", 
                                                      "Anxiety_Score", "Depression_Score", "Heart_Rate", 
                                                      "Suicidal_Thoughts"])

# Apply the same scaler transformation
sample_input_scaled = scaler.transform(sample_input_df)

# Predict
prediction = model.predict(sample_input_scaled)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

print(f"Predicted Diagnosis: {predicted_class[0]}")

