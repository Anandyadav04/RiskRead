import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

def train_model():
    print("Loading dataset...")
    
    # Try improved dataset first, fall back to original
    try:
        df = pd.read_csv("data/ingredients_improved.csv")
        print("Using improved dataset")
    except:
        df = pd.read_csv("data/ingredients.csv")
        print("Using original dataset")
    
    # Handle missing values
    df = df.dropna(subset=['ingredient', 'int_label'])
    
    X = df["ingredient"]
    y = df["int_label"]
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Class distribution:")
    for label in sorted(y.unique()):
        count = (y == label).sum()
        percentage = (count / len(y)) * 100
        label_name = ['Not Harmful', 'Controversial', 'Harmful'][label]
        print(f"  {label_name} (Label {label}): {count} samples ({percentage:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate class weights to handle imbalance
    classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    
    print(f"\nClass weights for balancing: {class_weight_dict}")
    
    # Create pipeline with balanced class weights
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=1500,
            stop_words='english',
            ngram_range=(1, 3),  # Include 1-3 word phrases
            min_df=2  # Ignore terms that appear only once
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight=class_weight_dict,  # Balanced weights
            C=0.5  # Regularization to prevent overfitting
        ))
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
    print("\n🧪 Test Predictions on Common Ingredients:")
    test_ingredients = [
        'cornflour', 'salt', 'onion', 'lemon', 'oil',
        'sugar', 'water', 'eggs', 'wheat flour'
    ]
    
    for ing in test_ingredients:
        pred = model.predict([ing])[0]
        label_name = ['Not Harmful', 'Controversial', 'Harmful'][pred]
        print(f"  {ing:15} -> {label_name}")
    
    # Save model
    with open("ml/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("\n✅ Model trained and saved as 'ml/model.pkl'")
    
    # Save test predictions for analysis
    test_results = pd.DataFrame({
        'ingredient': X_test,
        'true_label': y_test,
        'predicted_label': y_pred
    })
    test_results.to_csv("ml/test_predictions.csv", index=False)
    
    return model

if __name__ == "__main__":
    train_model()