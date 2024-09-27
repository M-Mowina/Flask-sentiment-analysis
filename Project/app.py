from flask import Flask, render_template, request, redirect, session, flash
from flask_session import Session
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
import bcrypt  
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
import regex as re
import emoji

# Database connection details (replace with your actual values)
DATABASE_FILE = 'SS.db'
# Download the 'stopwords' resource
nltk.download('stopwords')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
#Tokenize using BERT tokenizer (optional, replace with your desired tokenizer)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model3_path = 'amazon_full_model2.h5'
model2_path = 'final_model_V1.h5'
model1_path = 'podcasts_rnn_model.h5'
model = tf.keras.models.load_model(model3_path)

# Build the YouTube client
load_dotenv()
API_KEY = os.getenv("API_KEY")
youtube = build('youtube', 'v3', developerKey=API_KEY)


app = Flask(__name__)


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'GET':
        if not session.get('user_id'):
            return redirect('/login')
        
        return render_template("contact.html")
    else:
        #name = request.form.get('name')  # Optional, get name if provided
        email = request.form['email']
        message = request.form['message']
        print(message)
        # Connect to the database
        conn = sqlite3.connect(DATABASE_FILE)
        cur = conn.cursor()
        # Insert data into the ContactUs table
        try:
            user_id = session['user_id']
            name = session['user_name']
            # Use a parameterized query to prevent SQL injection attacks
            cur.execute("INSERT INTO ContactUs (user_id, name, email, message) VALUES (?, ?, ?, ?)", 
                  (user_id, name, email, message))
            conn.commit()
            respond = 'Your message has been sent successfully!'
            flash(respond, 'success')
            print(respond)
            return render_template('contact.html', error=respond)
        except sqlite3.Error as e:
            # Handle database errors gracefully (e.g., logging, flash message)
            error = str(e)
            flash(f"Database error: {error}", 'danger')
            print(error)
            return render_template('contact.html', error=error)
        finally:
            cur.close()
            conn.close()

    
@app.route('/text', methods=['GET', 'POST'])
def text_analysis():
    if request.method == 'GET':
        if not session.get('user_id'):
            return redirect('/login')
        
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
          error = str(e)
          flash(error, 'danger')
          return render_template('csv.html', error=error)

      # Check if there's a text column for prediction
      if 'text' not in df.columns:
          return render_template('csv.html', error="CSV must contain a 'text' column for prediction")

      # Get text data for prediction
      df['prediction'] = df['text'].apply(lambda text: predict_sentiment(text))
      # Create pie chart (same code as before)
      ax = df['prediction'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6, 4))
      ax.set_title('Count of Reviews For each lapel')

      # Customize the plot (optional)
      plt.legend(df['prediction'].unique())  # Add labels for pie slices

      # Save the plot to a byte buffer
      img_io = BytesIO()
      plt.savefig(img_io, format='png')
      img_io.seek(0)  # Reset the buffer to the beginning
      img_data = base64.b64encode(img_io.read()).decode('utf-8')  # Encode as base64 string

      # Improve table presentation (HTML styling)
      table_html = df.to_html(classes="data-table", escape=False)
      table_html = table_html.replace('table', '<table class="table table-striped">')  # Add Bootstrap classes

      # Return rendered template with results
      return render_template('csv_results.html', data=table_html, img_data=img_data)
    else:
      error = "No file selected"
      flash(error, 'danger')
      return render_template('csv.html', error=error)
  else:
    # Redirect to login page if user is not logged in
    if not session.get('user_id'):
            return redirect('/login')
    # Render upload page for GET requests
    return render_template('csv.html')


@app.route('/youtube', methods=['GET', 'POST'])
def youtube_analysis():
  if request.method == 'POST':
    # Get video ID from form
    link = request.form['video_id']
    video_id = extract_video_id(link)

    if video_id:
      # Build YouTube client
      youtube = build('youtube', 'v3', developerKey=API_KEY)

      # Get comments using your function
      comments = get_top_level_comments_for_video(youtube, video_id)

      # Check if comments were retrieved
      if comments:
        # Create DataFrame from comments
        comments_df = pd.DataFrame(comments)
        comments_df['prediction'] = comments_df['Comment'].apply(lambda text: predict_sentiment(text))
        comments_df = comments_df.drop(['Timestamp', 'VideoID'], axis=1)
        # Improve table presentation (HTML styling)
        table_html = comments_df.to_html(classes="data-table", escape=False)
        table_html = table_html.replace('table', '<table class="table table-striped">')  # Add Bootstrap classes


        # Create pie chart
        ax = comments_df['prediction'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6, 4))
        ax.set_title('Count of Reviews For each lapel')
        
        # Customize the plot (optional)
        plt.legend(comments_df['prediction'].unique())  # Add labels for pie slices

        # Save the plot to a byte buffer
        img_io = BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)  # Reset the buffer to the beginning
        img_data = base64.b64encode(img_io.read()).decode('utf-8')  # Encode as base64 string
        # Return rendered template with results
        return render_template('csv_results.html', data=table_html, img_data=img_data)
      else:
        # Handle case where no comments are found
        error = "No comments found for the provided video ID."
        flash(error, 'danger')
        return render_template('video_comments.html', error=error)
    else:
      # Handle case where no video ID is provided
      error = "Please provide a valid video ID."
      flash(error, 'danger')
      return render_template('video_comments.html', error=error)
  else:
    # Redirect to login page if user is not logged in
    if not session.get('user_id'):
            return redirect('/login')
    # Render form for GET requests
    return render_template('video_comments.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Hash the password before storing
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Connect to the database
        conn = sqlite3.connect(DATABASE_FILE)
        cur = conn.cursor()

        # Check for existing username or email
        try:
            cur.execute("SELECT * FROM Users WHERE username = ? OR email = ?", (username, email))
            existing_user = cur.fetchone()
            if existing_user:
                # Handle existing username or email error (e.g., flash message)
                error = "Username or email already exists."
                flash(error, 'danger')
                print(error)
                return render_template('register.html', error=error)
                
        except sqlite3.Error as e:
            # Handle database errors gracefully (e.g., logging, flash message)
            error = str(e)
            flash(error, 'danger')
            return render_template('register.html', error=f"Database error: {error}")

        # Insert new user into database
        try:
            cur.execute("INSERT INTO Users (username, password, email) VALUES (?, ?, ?)", (username, hashed_password, email))
            conn.commit()
            return redirect('/login')  # Redirect to login page after successful registration
        except sqlite3.Error as e:
            # Handle database errors gracefully (e.g., logging, flash message)
            error = str(e)
            flash(error, 'danger')
            return render_template('register.html', error=f"Database error: {error}")
        finally:
            cur.close()
            conn.close()

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Connect to the database
        conn = sqlite3.connect(DATABASE_FILE)
        cur = conn.cursor()

        # Check for user existence
        try:
            cur.execute("SELECT * FROM Users WHERE username = ?", (username,))
            user = cur.fetchone()
            if not user:
                # Handle invalid username error (e.g., flash message)
                error = "Invalid username or password."
                flash(error, 'danger')
                return render_template('login.html', error=error)
        except sqlite3.Error as e:
            # Handle database errors gracefully (e.g., logging, flash message)
            error = str(e)
            flash(error, 'danger')
            return render_template('login.html', error=f"Database error: {error}")

        # Verify password using bcrypt
        hashed_password = user[2]  # No decoding needed
        if not bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            # Handle invalid password error (e.g., flash message)
            error = "Invalid username or password."
            flash(error, 'danger')
            return render_template('login.html', error=error)

        session['user_id'] = user[0]
        session['user_name'] = user[1]
        print('session: ',session)
        print()
        print('user: ',user)
        return redirect("/")

    return render_template('login.html')


@app.route("/logout")
def logout():
    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")









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
  threshold = 0.35  # Adjust threshold based on your model's output range

  sentiment_label = 'Positive' if predicted_sentiment > threshold else 'Negative'

  return sentiment_label



def emoji_to_text(input_text):
    # Replace emojis with their textual representation
    text_with_emojis = emoji.demojize(input_text)
    return text_with_emojis

def preprocess_text(text):
  """Applies preprocessing steps to the given text."""
  # Lowercase text
  text = text.lower()

  # Convert emoji to text
  text = emoji_to_text(text)
  
  # Remove numbers (optional)
  text = re.sub('[0-9]+', '', text)

  # Remove special characters, punctuation including %, ., and ,
  text = re.sub(r"[^\w\s!@#\$*\(\)_\+=\^:\\]", " ", text)  # Preserve negation words

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

def extract_video_id(youtube_url):
    pattern = r"(?<=v=|\/videos\/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v=|&v=|%2Fvideos%2F|embed%2F|%2Fv%2F|%2Fe%2F|%2Fwatch%3Fv%3D|%26v%3D|%3Fv%3D)([\w-]+)"
    match = re.search(pattern, youtube_url)
    if match:
        return match.group(1)
    else:
        return None

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

class User:
    def __init__(self, user_id, username, email):
        self.user_id = user_id
        self.username = username
        self.email = email


if __name__ == '__main__':
    app.run()
