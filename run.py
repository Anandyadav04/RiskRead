#!/usr/bin/env python3
"""
RiskRead - Main Application Launcher
"""
import os
import sys
import subprocess
import webbrowser
from time import sleep

def check_dependencies():
    """Check if all required packages are installed"""
    required = ['flask', 'pandas', 'scikit-learn', 'opencv-python', 'pytesseract', 'nltk']
    
    print("Checking dependencies...")
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} is missing")
            return False
    return True

def setup_project():
    """Setup project structure and train model"""
    print("\nSetting up RiskRead...")
    
    # Download dataset if not exists
    if not os.path.exists("data/ingredients.csv"):
        print("Downloading dataset...")
        from datasets import load_dataset
        import pandas as pd
        
        dataset = load_dataset("foodvisor-nyu/labeled-food-ingredients")
        df = dataset["train"].to_pandas()
        df.to_csv("data/ingredients.csv", index=False)
        print(f"Dataset saved with {len(df)} samples")
    
    # Train model if not exists
    if not os.path.exists("ml/model.pkl"):
        print("Training ML model...")
        import pickle
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        import pandas as pd
        
        df = pd.read_csv("data/ingredients.csv")
        df = df.dropna(subset=['ingredient', 'int_label'])
        
        X = df["ingredient"]
        y = df["int_label"]
        
        model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=1000)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        model.fit(X, y)
        
        with open("ml/model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        print("Model trained and saved")
    
    print("\n✅ Setup complete!")

def main():
    """Main function to run the application"""
    print("=" * 60)
    print("       RISKREAD - Ingredient Safety Analyzer")
    print("=" * 60)
    
    if not check_dependencies():
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")
        return
    
    setup_project()
    
    print("\n" + "=" * 60)
    print("Starting RiskRead Application...")
    print("The web interface will open in your browser shortly.")
    print("=" * 60 + "\n")
    
    # Start Flask app
    from app.app import app
    
    # Open browser after delay
    def open_browser():
        sleep(2)
        webbrowser.open('http://localhost:5000')
    
    import threading
    threading.Thread(target=open_browser).start()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

if __name__ == "__main__":
    main()