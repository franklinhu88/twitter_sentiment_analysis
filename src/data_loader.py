import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources (run once)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
DATA_PATH = './data/twitter_training.csv'  # Adjust the path if needed

# Column names as provided
column_names = ["Tweet ID", "entity", "sentiment", "Tweet content"]

def load_data():
    # Read only the first 1,000 rows of the dataset
    df = pd.read_csv(DATA_PATH, names=column_names, header=None, nrows=1000)  # Add nrows=10000
    return df



# Clean text function
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove non-alphabetic characters (punctuation, numbers)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back to string
    return ' '.join(tokens)

# Main function
def preprocess_data():
    df = load_data()
    
    # Check the columns in the dataset
    print("Columns in the dataset:", df.columns)
    
    if df['Tweet content'].isnull().any():
        df['Tweet content'] = df['Tweet content'].fillna('') # Replace NaN with empty string
    
    # Apply the clean_text function
    df['Tweet content'] = df['Tweet content'].apply(clean_text)
    
    return df


if __name__ == "__main__":
    df = preprocess_data()
    print(df.head())  # Check the cleaned data
