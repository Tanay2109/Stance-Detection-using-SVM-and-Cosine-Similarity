import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Define preprocessing functions
def clean_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = ''.join([c for c in text if c.isalpha() or c.isspace()])
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        # Rejoin
        return ' '.join(tokens)
    else:
        return ""

# Main function to run the entire process
def main():
    print("Fake News Detection using SVM")
    print("Loading the FNC1 dataset...")
    
    try:
        # Load the dataset
        train_bodies = pd.read_csv("fnc-1/train_bodies.csv")
        train_stances = pd.read_csv("fnc-1/train_stances.csv")
        
        print("Dataset loaded successfully!")
        print(f"Number of bodies: {len(train_bodies)}")
        print(f"Number of stances: {len(train_stances)}")
        
        # Check for missing values
        print("\nMissing values in bodies:", train_bodies.isnull().sum())
        print("Missing values in stances:", train_stances.isnull().sum())
        
        # Check stance distribution
        stance_distribution = train_stances['Stance'].value_counts()
        print("\nStance distribution:")
        print(stance_distribution)
        
        # Visualize stance distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=stance_distribution.index, y=stance_distribution.values)
        plt.title('Distribution of Stances in FNC1 Dataset')
        plt.ylabel('Count')
        plt.xlabel('Stance')
        plt.savefig('stance_distribution.png')
        print("Stance distribution visualization saved as 'stance_distribution.png'")
        
        # Merge the datasets
        print("\nMerging datasets...")
        df = pd.merge(train_stances, train_bodies, on='Body ID')
        
        # Preprocess the data
        print("Preprocessing data...")
        df['clean_headline'] = df['Headline'].apply(clean_text)
        df['clean_body'] = df['articleBody'].apply(clean_text)
        
        # Create TF-IDF features
        print("Extracting features using TF-IDF...")
        
        # Initialize a single TF-IDF vectorizer for both headline and body
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        
        # Fit the vectorizer on both headline and body text to create a unified vocabulary
        tfidf_vectorizer.fit(df['clean_headline'].tolist() + df['clean_body'].tolist())
        
        # Transform headline and body text using the same fitted vectorizer
        headline_features = tfidf_vectorizer.transform(df['clean_headline'])
        body_features = tfidf_vectorizer.transform(df['clean_body'])
        
        print("Calculating similarity features...")
        # Calculate similarity features
        similarity_features = []
        for i in range(len(df)):
            headline_vec = headline_features[i].toarray().flatten()
            body_vec = body_features[i].toarray().flatten()
            
            # Now both vectors have the same dimensions, so cosine_similarity will work
            cos_sim = cosine_similarity(headline_vec.reshape(1, -1), body_vec.reshape(1, -1))[0][0]
            
            # Count of common words
            headline_words = set(df['clean_headline'].iloc[i].split())
            body_words = set(df['clean_body'].iloc[i].split())
            common_word_count = len(headline_words.intersection(body_words))
            
            similarity_features.append([cos_sim, common_word_count])
        
        # Convert to numpy array
        similarity_features = np.array(similarity_features)
        
        # Combine the features
        from scipy.sparse import hstack
        X = hstack([headline_features, body_features, similarity_features])
        
        # Create target variable
        y = df['Stance']
        
        # Convert stance labels to numeric
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print("Feature matrix shape:", X.shape)
        print("Target variable shape:", y_encoded.shape)
        
        # Split the data
        print("\nSplitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print("Training set size:", X_train.shape[0])
        print("Testing set size:", X_test.shape[0])
        
        # Train the SVM model
        print("\nTraining the SVM model...")
        svm_classifier = svm.SVC(kernel='linear', C=1.0, random_state=42)
        svm_classifier.fit(X_train, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = svm_classifier.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_, 
                    yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix visualization saved as 'confusion_matrix.png'")
        
        # Hyperparameter tuning
        print("\nPerforming hyperparameter tuning (this may take a while)...")
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        
        grid_search = GridSearchCV(
            svm.SVC(random_state=42), 
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best parameters and score
        print("\nBest parameters:", grid_search.best_params_)
        print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
        
        # Evaluate best model
        best_svm = grid_search.best_estimator_
        y_pred_best = best_svm.predict(X_test)
        best_accuracy = accuracy_score(y_test, y_pred_best)
        print(f"Test accuracy with best parameters: {best_accuracy:.4f}")
        
        # Implement the FNC-1 scoring function
        def fnc_score(y_true, y_pred, classes):
            y_true = label_encoder.inverse_transform(y_true)
            y_pred = label_encoder.inverse_transform(y_pred)
            
            # Calculate related vs unrelated score (25% weight)
            y_true_binary = np.array([0 if s == 'unrelated' else 1 for s in y_true])
            y_pred_binary = np.array([0 if s == 'unrelated' else 1 for s in y_pred])
            binary_score = accuracy_score(y_true_binary, y_pred_binary)
            
            # Calculate agrees/disagrees/discusses score (75% weight)
            # Extract indices of related articles
            related_indices = np.where(y_true_binary == 1)[0]
            if len(related_indices) > 0:
                y_true_related = [y_true[i] for i in related_indices]
                y_pred_related = [y_pred[i] for i in related_indices]
                related_score = accuracy_score(y_true_related, y_pred_related)
            else:
                related_score = 0
            
            # Calculate weighted score
            weighted_score = 0.25 * binary_score + 0.75 * related_score
            return weighted_score
        
        # Evaluate using FNC-1 metric
        fnc_weighted_score = fnc_score(y_test, y_pred_best, label_encoder.classes_)
        print(f"FNC-1 weighted score: {fnc_weighted_score:.4f}")
        
        # Save model and vectorizers
        print("\nSaving model and preprocessing components...")
        with open('svm_fnc1_model.pkl', 'wb') as f:
            pickle.dump(best_svm, f)
        
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print("Model and preprocessing components saved successfully!")
        
        # Define prediction function
        def predict_stance(headline, article_body):
            # Preprocess
            clean_headline = clean_text(headline)
            clean_body = clean_text(article_body)
            
            # Transform headline and body
            headline_features = tfidf_vectorizer.transform([clean_headline])
            body_features = tfidf_vectorizer.transform([clean_body])
            
            # Calculate similarity features
            headline_vec = headline_features.toarray().flatten()
            body_vec = body_features.toarray().flatten()
            cos_sim = cosine_similarity(headline_vec.reshape(1, -1), body_vec.reshape(1, -1))[0][0]
            
            headline_words = set(clean_headline.split())
            body_words = set(clean_body.split())
            common_word_count = len(headline_words.intersection(body_words))
            
            similarity_features = np.array([[cos_sim, common_word_count]])
            
            # Combine features
            X = hstack([headline_features, body_features, similarity_features])
            
            # Predict
            prediction = best_svm.predict(X)
            stance = label_encoder.inverse_transform(prediction)[0]
            
            return stance
        
        # Test the prediction function
        print("\nTesting the prediction function with a sample:")
        test_headline = "Climate change is a hoax"
        test_body = "Scientists worldwide agree that climate change is real and caused by human activities."
        predicted_stance = predict_stance(test_headline, test_body)
        print(f"Headline: {test_headline}")
        print(f"Body: {test_body}")
        print(f"Predicted stance: {predicted_stance}")
        
        print("\nFake News Detection model training and evaluation completed successfully!")
        
    except FileNotFoundError:
        print("Error: Dataset files not found. Please make sure the FNC1 dataset is in the 'fnc-1' directory.")
        print("You can download it from: https://github.com/FakeNewsChallenge/fnc-1")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
