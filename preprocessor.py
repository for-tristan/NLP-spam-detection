import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    """Shared preprocessing function for all team members"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URLs
    text = re.sub(r'\d+', '', text)  # Numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Punctuation
    
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [stemmer.stem(w) for w in tokens]
    
    return " ".join(tokens)

def load_and_prepare_data(file_path="spam.csv"):
    """Load and prepare dataset"""
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean_text'] = df['text'].apply(preprocess)
    return df