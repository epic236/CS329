import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. LOAD SPLIT DATASETS ---
# Replace paths with your actual local file locations
train_df = pd.read_csv("datasets/dart_ready_train.csv")
test_df = pd.read_csv("datasets/dart_ready_test.csv")
dev_df = pd.read_csv("datasets/dart_ready_dev.csv") # Optional: use for validation during tuning

target_col = 'dialect'
text_col = 'text'

# --- 2. ARABIC NORMALIZATION ---
def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[\u064B-\u0652]", "", text) 
    text = re.sub(r"[أإآ]", "ا", text)           
    text = re.sub(r"ى", "ي", text)              
    text = re.sub(r"ة", "ه", text)              
    text = re.sub(r"[^\u0621-\u064A\s]", " ", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- 3. PIPELINE ---
# Apply normalization to all sets
train_df['text_clean'] = train_df[text_col].apply(normalize_arabic)
test_df['text_clean'] = test_df[text_col].apply(normalize_arabic)

# Use the pre-split data
X_train = train_df['text_clean']
y_train = train_df[target_col]
X_test = test_df['text_clean']
y_test = test_df[target_col]

# Vectorization (Fit ONLY on training data)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=25000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# --- 4. EVALUATION ---
y_pred = model.predict(X_test_tfidf)
labels = sorted(y_test.unique())

print(f"Overall Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nRegion-Level Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Purples')
plt.title('DART Dataset Region Confusion Matrix')
plt.ylabel('Actual Region')
plt.xlabel('Predicted Region')
plt.show()

# Dialect detection function remains the same
def detect_dialect(text: str) -> str:
    text_clean = normalize_arabic(text)
    text_vectorized = vectorizer.transform([text_clean])
    return model.predict(text_vectorized)[0]

# NEW: checks whether the input is mostly Arabic script
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")

# NEW: checks whether a character is Arabic script
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")

def detect_dialect(text: str) -> str:
    text = str(text)

    # NEW: reject inputs that are mostly not Arabic script
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "NON_ARABIC"

    arabic_letters = sum(bool(ARABIC_CHAR_RE.match(ch)) for ch in letters)
    arabic_ratio = arabic_letters / len(letters)

    # NEW: if too little Arabic is present, do not force a dialect prediction
    if arabic_ratio < 0.35:
        return "NON_ARABIC"

    text_clean = normalize_arabic(text)

    # NEW: if normalization strips everything out, treat it as non-Arabic
    if not text_clean:
        return "NON_ARABIC"

    text_vectorized = vectorizer.transform([text_clean])
    return model.predict(text_vectorized)[0]