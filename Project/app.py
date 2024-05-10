from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
from transformers import BertTokenizer
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download the 'stopwords' resource
nltk.download('stopwords')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
#Tokenize using BERT tokenizer (optional, replace with your desired tokenizer)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = tf.keras.models.load_model('amazon_full_model2.h5')


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('site.html')
    
    else:
        text = request.form['text']
        print(text)
        result = predict_sentiment(text)
        print(result)
        return render_template('site.html', result=result)

def predict_sentiment(text, model = model, tokenizer = tokenizer, max_len = 200):
  """Predicts sentiment for a given text using the provided model and tokenizer.

  Args:
      text: The text to predict sentiment for (string).
      model: The trained sentiment analysis model.
      tokenizer: The tokenizer used to preprocess the text.
      max_len: The maximum sequence length for the model (integer).

  Returns:
      A tuple containing:
          - predicted_sentiment: The predicted sentiment score (float).
          - sentiment_label: The sentiment label based on a threshold (string).
  """

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
  """Applies preprocessing steps to the given text."""
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

if __name__ == '__main__':
    app.run()
