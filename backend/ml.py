# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
import re
import nltk
import joblib
import os
from langdetect import detect
from googletrans import Translator

# Download stopwords
nltk.download('stopwords')

# Load the data
data = pd.read_csv('Symptom2Disease.csv')

# Data Cleaning
data.dropna(inplace=True)  # Remove missing values

# Initialize custom Hindi stopwords
hindi_stopwords = set([
    'और', 'है', 'के', 'को', 'में', 'से', 'यह', 'कि', 'पर', 'नहीं', 
    'तो', 'किया', 'होगा', 'होगी', 'होगे', 'क्योंकि', 'अगर', 'लेकिन', 
    'जब', 'तब', 'साथ', 'भी', 'या', 'अभी', 'किसी', 'किस', 'उन', 
    'उनका', 'उनकी', 'हम', 'हमारा', 'हमारी', 'आप', 'आपका', 'आपकी', 
    'यहाँ', 'वहाँ', 'कहाँ', 'कब', 'क्यों', 'कैसे', 'क्या', 'कौन', 
    'जैसे'
])

# Multilingual stop words
stop_words = {
    'en': set(stopwords.words('english')),
    'hi': hindi_stopwords,
    # Add more languages as needed
}

# Initialize the translator
translator = Translator()

# Text Preprocessing Function
def preprocess_text(text):
    # Detect language
    lang = detect(text)
    lang = lang if lang in stop_words else 'en'  # Default to English if language is not supported

    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [word for word in text if word not in stop_words[lang]]  # Remove stop words
    return ' '.join(text)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Feature Selection
X = data['text']  # Features
y = data['label']  # Target variable

# Text Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'symptom_disease_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')  # Save the vectorizer for later use

# Function to save new disease inputs and update Excel
def save_new_disease(symptoms, disease_name):
    # Save to text file
    with open('new_diseases.txt', 'a') as file:
        file.write(f"{symptoms},{disease_name}\n")
    
    # Save to Excel
    new_data = pd.DataFrame({'text': [symptoms], 'label': [disease_name]})
    excel_file = 'Symptom2Disease.xlsx'
    
    if os.path.exists(excel_file):
        # Append to existing Excel file
        with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            new_data.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        # Create a new Excel file
        new_data.to_excel(excel_file, index=False)

# Function to validate user input
def is_valid_input(symptoms):
    # Check for empty input
    if not symptoms.strip():
        return False
    # Preprocess the input to check for meaningful content
    processed_symptoms = preprocess_text(symptoms)
    # Check if the processed symptoms contain meaningful words
    if len(processed_symptoms.split()) < 2:  # Require at least 2 meaningful words
        return False
    return True

# Function to predict disease based on user input
def predict_disease(symptoms, top_n=3):
    # Load the model and vectorizer
    model = joblib.load('symptom_disease_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    
    # Detect language and translate if necessary
    lang = detect(symptoms)
    if lang == 'hi':  # If input is in Hindi
        symptoms = translator.translate(symptoms, src='hi', dest='en').text

    # Validate input
    if not is_valid_input(symptoms):
        return "Error: Please enter valid symptoms. Ensure your input is meaningful and contains at least two words."

    # Preprocess the input symptoms
    processed_symptoms = preprocess_text(symptoms)

    # Vectorize the input symptoms
    symptoms_vectorized = vectorizer.transform([processed_symptoms])
    
    # Get prediction probabilities
    prediction_probs = model.predict_proba(symptoms_vectorized)
    
    # Get the indices of the top N predictions
    top_n_indices = prediction_probs[0].argsort()[-top_n:][::-1]
    
    # Get the corresponding disease names and their probabilities
    top_diseases = [(model.classes_[i], prediction_probs[0][i]) for i in top_n_indices]
    
    return top_diseases

# Function to retrain the model with new data
def retrain_model():
    # Load new data from the file
    if os.path.exists('new_diseases.txt'):
        new_data = pd.read_csv('new_diseases.txt', names=['text', 'label'])
        # Append new data to the existing dataset
        global data  # Ensure we are modifying the global data variable
        data = pd.concat([data, new_data], ignore_index=True)
        
        # Proceed with the existing training steps
        data.dropna(inplace=True)  # Remove missing values
        data['text'] = data['text'].apply(preprocess_text)
        X = data['text']
        y = data['label']
        X_vectorized = vectorizer.fit_transform(X)  # Refit vectorizer with new data
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        
        # Train the model again
        model.fit(X_train, y_train)
        
        # Save the updated model
        joblib.dump(model, 'symptom_disease_model.pkl')
        print("Model retrained with new data.")

# User Input Section
if __name__ == "__main__":
    user_symptoms = input("Please enter your symptoms: ")
    predicted_diseases = predict_disease(user_symptoms)
    
    if isinstance(predicted_diseases, str):  # Check if an error message was returned
        print(predicted_diseases)
    else:
        print("Predicted diseases and their probabilities:")
        for disease, prob in predicted_diseases:
            print(f"{disease}: {prob:.2f}")

# Call retrain_model() periodically or based on a specific condition.