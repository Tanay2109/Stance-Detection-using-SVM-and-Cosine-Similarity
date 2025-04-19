# Load saved components
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy.sparse import hstack
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and components
with open(r'SVM\svm_fnc1_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open(r'SVM\tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

with open(r'SVM\label_encoder.pkl', 'rb') as f:
    loaded_encoder = pickle.load(f)

# Define preprocess function
def clean_text(text):
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

    if isinstance(text, str):
        text = text.lower()
        text = ''.join([c for c in text if c.isalpha() or c.isspace()])
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in stopwords.words('english')]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
        return ' '.join(tokens)
    return ""

# Load dataset again
df = pd.read_csv("fnc-1/train_stances.csv").merge(pd.read_csv("fnc-1/train_bodies.csv"), on="Body ID")

# Sample 10 examples
samples = df.sample(10, random_state=42).reset_index(drop=True)

# Predict and compare
print("\n🔍 Real FNC-1 Dataset Predictions:\n")
for i, row in samples.iterrows():
    headline = row['Headline']
    body = row['articleBody']
    actual = row['Stance']

    # Clean
    clean_head = clean_text(headline)
    clean_body = clean_text(body)

    # Transform
    head_feat = loaded_vectorizer.transform([clean_head])
    body_feat = loaded_vectorizer.transform([clean_body])

    # Similarity features
    cos_sim = cosine_similarity(head_feat.toarray(), body_feat.toarray())[0][0]
    common_words = len(set(clean_head.split()).intersection(set(clean_body.split())))
    sim_feats = np.array([[cos_sim, common_words]])

    # Combine
    x_input = hstack([head_feat, body_feat, sim_feats])

    # Predict
    pred_encoded = loaded_model.predict(x_input)
    pred_label = loaded_encoder.inverse_transform(pred_encoded)[0]

    print(f"{i+1}. Headline: {headline}")
    print(f"   Body: {body[:100]}...")  # Print first 100 chars
    print(f"   ✅ Actual Stance: {actual}")
    print(f"   🤖 Predicted Stance: {pred_label}")
    print("-" * 80)


# Load test data
test_df = pd.read_csv("fnc-1/test_stances_unlabeled.csv").merge(
    pd.read_csv("fnc-1/test_bodies.csv"), on="Body ID"
)

# Sample 10 examples
test_samples = test_df.sample(10, random_state=42).reset_index(drop=True)

print("\n🤖 Predictions on test_stances_unlabeled.csv:\n")
for i, row in test_samples.iterrows():
    headline = row['Headline']
    body = row['articleBody']

    # Clean text
    clean_head = clean_text(headline)
    clean_body = clean_text(body)

    # Vectorize
    head_feat = loaded_vectorizer.transform([clean_head])
    body_feat = loaded_vectorizer.transform([clean_body])

    # Similarity features
    cos_sim = cosine_similarity(head_feat.toarray(), body_feat.toarray())[0][0]
    common_words = len(set(clean_head.split()).intersection(set(clean_body.split())))
    sim_feats = np.array([[cos_sim, common_words]])

    # Combine
    x_input = hstack([head_feat, body_feat, sim_feats])

    # Predict
    pred_encoded = loaded_model.predict(x_input)
    pred_label = loaded_encoder.inverse_transform(pred_encoded)[0]

    print(f"{i+1}. Headline: {headline}")
    print(f"   Body: {body[:100]}...")  # Print partial body
    print(f"   🤖 Predicted Stance: {pred_label}")
    print("-" * 80)
