import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# =========================
# Download stopwords
# =========================
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# =========================
# Preprocessing functions
# =========================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)

# =========================
# Load dataset
# =========================
df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# =========================
# Preprocess data
# =========================
df['clean_text'] = df['text'].apply(preprocess)

# =========================
# Split data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# =========================
# Vectorization
# =========================
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# Model
# =========================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test_vec)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📌 Report:\n", classification_report(y_test, y_pred))

# =========================
# Interactive testing
# =========================
print("\n💬 Test your messages (type 'exit' to stop)\n")

while True:
    text = input("Enter message: ")

    if text.lower() == "exit":
        break

    processed = preprocess(text)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)

    print("👉 Result:", "Spam 🚨" if pred[0] == 1 else "Ham ✅")