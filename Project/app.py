from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
from transformers import BertTokenizer
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pandas as pd

# Download the 'stopwords' resource
nltk.download('stopwords')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
#Tokenize using BERT tokenizer (optional, replace with your desired tokenizer)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = tf.keras.models.load_model('amazon_full_model2.h5')


app = Flask(__name__)


@app.route("/")
def analysis():
    return render_template("analysis.html")


@app.route('/text', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('text.html')
    
    else:
        text = request.form['text']
        print(text)
        result = predict_sentiment(text)
        print(result)
        return render_template('text.html', result=result)
    
    
@app.route('/csv', methods=['GET', 'POST'])
def csv_file():
  if request.method == 'POST':
    # Get uploaded file
    uploaded_file = request.files['csvfile']
    if uploaded_file.filename != '':
      # Read CSV data into a Pandas DataFrame
      df = pd.read_csv(uploaded_file)
      
      # Check if there's a text column for prediction
      if 'text' not in df.columns:
          return render_template('csv.html', error="CSV must contain a 'text' for prediction")
      
      # Get text data for prediction
      df['prediction'] = df['text'].apply(lambda text: predict_sentiment(text))
      
      # Make predictions using your model
      predictions = df['prediction']
      
      # Return rendered template with results
      return render_template('csv.html', data=df.to_html())
    else:
      return render_template('csv.html', error="No file selected")
  else:
    # Render upload page for GET requests
    return render_template('csv.html')

















def predict_sentiment(text, model = model, tokenizer = tokenizer, max_len = 200):
  
  # Preprocess the text
  preprocessed_text = preprocess_text(text)  # Replace with your preprocessing function

  # Tokenize the text
  tokens = tokenizer.tokenize(preprocessed_text, padding='max_length', truncation=True)

  # Convert tokens to IDs
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # Pad the sequence (if model requires it)
  padded_input = pad_sequences([input_ids], maxlen=max_len)

  # Make the prediction
  predicted_sentiment = model.predict(padded_input)[0][0]  # Assuming single output

  # Define a threshold for sentiment classification (optional)
  threshold = 0.5  # Adjust threshold based on your model's output range

  sentiment_label = 'Positive' if predicted_sentiment > threshold else 'Negative'

  return sentiment_label

def preprocess_text(text):
  # Lowercase text
  text = text.lower()

  # Remove numbers (optional)
  text = re.sub('[0-9]+', '', text)  # Consider keeping numbers for specific domains

  # Remove special characters and some punctuation
  text = re.sub(r"[^\w\s!@#\$%&*\(\)_\+=\^:\.,;]", " ", text)  # Preserve negation words, some punctuation
  
  # Lemmatization (preferred)
  text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

  # Clean URLs
  text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)

  # Clean Emails
  text = re.sub('@[^\s]+', ' ', text)
  
  # Stop word removal (optional)
  text = ' '.join([word for word in text.split() if word not in stop_words])

  return text

def predict_csv(csv_file):
      df = pd.read_csv(csv_file)
      
      # Check if there's a text column for prediction
      if 'text' not in df.columns:
          return "CSV must contain a 'text' for prediction"
      
      # Add predictions as a new column in the DataFrame
      df['prediction'] = df['text'].apply(lambda text: predict_sentiment(text))
      
      # Return rendered template with results
      return df['prediction']


if __name__ == '__main__':
    app.run()
