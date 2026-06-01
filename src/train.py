import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from utils import clean_resume

def train_pipeline():
    print("Loading dataset...")
    df = pd.read_csv("data/UpdatedResumeDataSet.csv")
    
    print("Preprocessing text corpuses...")
    df["Cleaned_Resume"] = df["Resume"].apply(clean_resume)
    
    X = df["Cleaned_Resume"]
    y = df["Category"]
    
    # Stratified split to handle minor class imbalances
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Vectorizing using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2500)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Multinomial Naive Bayes Classifier...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)
    
    # Evaluation metrics
    preds = model.predict(X_test_vec)
    print("\n--- Model Evaluation Matrix ---")
    print(classification_report(y_test, preds))
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(model, "models/resume_classifier.pkl")
    print("Artifacts successfully serialized to disk inside 'models/'.")

if __name__ == "__main__":
    train_pipeline()