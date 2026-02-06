import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ingredients_improved.csv")

print("=== Dataset Analysis ===")
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Check label distribution
if 'int_label' in df.columns:
    print("\nLabel distribution:")
    print(df['int_label'].value_counts())
    
    # Create simple visualization
    plt.figure(figsize=(10, 6))
    df['int_label'].value_counts().plot(kind='bar')
    plt.title('Distribution of Ingredient Safety Labels')
    plt.xlabel('Label (0=Not Harmful, 1=Controversial, 2=Harmful)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('data/label_distribution.png')
    print("Distribution chart saved as 'data/label_distribution.png'")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Show sample ingredients
print("\nSample ingredients:")
for i in range(5):
    print(f"{i+1}. {df.iloc[i]['ingredient']} -> Label: {df.iloc[i]['int_label']}")