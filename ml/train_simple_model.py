import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def train_simple_model():
    print("Training SIMPLE model...")
    
    # Try balanced dataset first
    try:
        df = pd.read_csv("data/balanced_ingredients.csv")
        print("Using balanced dataset")
    except:
        df = pd.read_csv("data/ingredients.csv")
        print("Using original dataset")
    
    df = df.dropna(subset=['ingredient', 'int_label'])
    
    X = df["ingredient"]
    y = df["int_label"]
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(X)}")
    
    # Ensure we have all 3 classes
    if len(y.unique()) < 3:
        print("WARNING: Dataset missing some classes!")
        print(f"Classes present: {sorted(y.unique())}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SIMPLE MODEL - Naive Bayes works better with small data
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=500,  # Smaller for simpler model
            stop_words='english',
            ngram_range=(1, 2)
        )),
        ("clf", MultinomialNB(alpha=0.1))  # Naive Bayes with smoothing
    ])
    
    # Train
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Not Harmful', 'Controversial', 'Harmful']))
    
    # Test on common ingredients
    print("\n🧪 TEST PREDICTIONS - Should be NOT HARMFUL:")
    test_safe = [
        'cornflour', 'salt', 'onion', 'lemon', 'oil',
        'sugar', 'water', 'eggs', 'wheat flour',
        'milk', 'butter', 'rice', 'bread', 'cheese'
    ]
    
    correct = 0
    total = len(test_safe)
    
    for ing in test_safe:
        try:
            pred = model.predict([ing])[0]
            label_name = ['Not Harmful', 'Controversial', 'Harmful'][pred]
            
            if pred == 0:
                print(f"  ✅ {ing:15} -> {label_name}")
                correct += 1
            else:
                print(f"  ❌ {ing:15} -> {label_name} (SHOULD BE NOT HARMFUL)")
        except:
            print(f"  ⚠️ {ing:15} -> Error in prediction")
    
    print(f"\n✅ {correct}/{total} correct safe predictions")
    
    # Test on harmful ingredients
    print("\n🧪 TEST PREDICTIONS - Should be HARMFUL:")
    test_harmful = [
        'aspartame', 'sodium benzoate', 'trans fat',
        'bha', 'bht', 'red 40'
    ]
    
    for ing in test_harmful:
        try:
            pred = model.predict([ing])[0]
            label_name = ['Not Harmful', 'Controversial', 'Harmful'][pred]
            
            if pred == 2:
                print(f"  ✅ {ing:20} -> {label_name}")
            else:
                print(f"  ❌ {ing:20} -> {label_name} (SHOULD BE HARMFUL)")
        except:
            print(f"  ⚠️ {ing:20} -> Error in prediction")
    
    # Save model
    with open("ml/model_simple.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("\n✅ Simple model trained and saved as 'ml/model_simple.pkl'")
    
    # Also update the main model
    with open("ml/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("✅ Also saved as 'ml/model.pkl' (main model)")
    
    return model

if __name__ == "__main__":
    train_simple_model()