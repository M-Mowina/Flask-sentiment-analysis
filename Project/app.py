from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pandas as pd
from dotenv import load_dotenv # type: ignore
import os
from googleapiclient.discovery import build # type: ignore
import pandas as pd
import getpass

# Download the 'stopwords' resource
nltk.download('stopwords')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
#Tokenize using BERT tokenizer (optional, replace with your desired tokenizer)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = tf.keras.models.load_model('amazon_full_model2.h5')
# Build the YouTube client
load_dotenv()
API_KEY = os.getenv("API_KEY")
youtube = build('youtube', 'v3', developerKey=API_KEY)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("analysis.html")


@app.route('/text', methods=['GET', 'POST'])
def text_analysis():
    if request.method == 'GET':
        return render_template('text.html')
    
    else:
        text = request.form['text']
        print(text)
        result = predict_sentiment(text)
        print(result)
        return render_template('text.html', result=result)
    
    
@app.route('/csv', methods=['GET', 'POST'])
def csv_analysis():
  if request.method == 'POST':
    # Get uploaded file
    uploaded_file = request.files['csvfile']
    if uploaded_file.filename != '':
      # Read CSV data into a Pandas DataFrame
      try:
          df = pd.read_csv(uploaded_file)
      except Exception as e:
          # Handle potential errors during CSV reading
          return render_template('csv.html', error=f"Error reading CSV: {str(e)}")

      # Check if there's a text column for prediction
      if 'text' not in df.columns:
          return render_template('csv.html', error="CSV must contain a 'text' column for prediction")

      # Get text data for prediction
      df['prediction'] = df['text'].apply(lambda text: predict_sentiment(text))

      # Improve table presentation (HTML styling)
      table_html = df.to_html(classes="data-table", escape=False)
      table_html = table_html.replace('table', '<table class="table table-striped">')  # Add Bootstrap classes

      # Return rendered template with results
      return render_template('csv_results.html', data=table_html)
    else:
      return render_template('csv.html', error="No file selected")
  else:
    # Render upload page for GET requests
    return render_template('csv.html')


@app.route('/youtube', methods=['GET', 'POST'])
def youtube_analysis():
  if request.method == 'POST':
    # Get video ID from form
    video_id = request.form['video_id']

    if video_id:
      # Build YouTube client
      youtube = build('youtube', 'v3', developerKey=API_KEY)

      # Get comments using your function
      comments = get_top_level_comments_for_video(youtube, video_id)

      # Check if comments were retrieved
      if comments:
        # Create DataFrame from comments
        comments_df = pd.DataFrame(comments)
        comments_df['Comment'] = comments_df['Comment'].apply(lambda text: predict_sentiment(text))
        comments_df = comments_df.drop(['Timestamp', 'VideoID'], axis=1)
        # Improve table presentation (HTML styling)
        table_html = comments_df.to_html(classes="data-table", escape=False)
        table_html = table_html.replace('table', '<table class="table table-striped">')  # Add Bootstrap classes

        # Return rendered template with results
        return render_template('csv_results.html', data=table_html)
      else:
        # Handle case where no comments are found
        return render_template('video_comments.html', error="No comments found for the provided video ID.")
    else:
      # Handle case where no video ID is provided
      return render_template('video_comments.html', error="Please enter a valid YouTube video ID.")
  else:
    # Render form for GET requests
    return render_template('video_comments.html')














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

def get_top_level_comments_for_video(youtube, video_id):
  """
  Fetches top-level comments (without replies) for a single video.

  Args:
      youtube: Authorized YouTube Data API v3 service object.
      video_id: ID of the video for which comments are desired.

  Returns:
      A list of dictionaries, where each dictionary represents a comment
      with details like timestamp, username, comment text, etc.
  """
  top_level_comments = []
  next_page_token = None

  while True:
    comment_request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        pageToken=next_page_token,
        textFormat="plainText",
        maxResults=100
    )
    comment_response = comment_request.execute()

    for item in comment_response['items']:
      top_comment = item['snippet']['topLevelComment']['snippet']
      top_level_comments.append({
          'Timestamp': top_comment['publishedAt'],
          'Username': top_comment['authorDisplayName'],
          'VideoID': video_id,
          'Comment': top_comment['textDisplay'],
          'Date': top_comment['updatedAt'] if 'updatedAt' in top_comment else top_comment['publishedAt']
      })

    next_page_token = comment_response.get('nextPageToken')
    if not next_page_token or len(top_level_comments) >= 100:  # Stop when 100 comments are collected
      break

  return top_level_comments


if __name__ == '__main__':
    app.run()
