# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# ==============================
# 3. Preprocessing
# ==============================
# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Remove missing values
df.dropna(inplace=True)

# ==============================
# 4. Split Data (Train/Test)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# ==============================
# 5. Build Pipeline Model
# ==============================
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', MultinomialNB(alpha=0.5))
])

# ==============================
# 6. Train Model
# ==============================
model.fit(X_train, y_train)

# ==============================
# 7. Prediction
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 8. Evaluation
# ==============================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==============================
# 9. Test Custom Message
# ==============================
def predict_message(msg):
    prediction = model.predict([msg])[0]
    return "Spam" if prediction == 1 else "Ham"

# ==============================
# 10. Try Example
# ==============================
print("\nCustom Test:")
print(predict_message("Congratulations! You've won a free ticket. Call now!"))
