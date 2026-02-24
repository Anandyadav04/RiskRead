from datasets import load_dataset
import pandas as pd

# Load dataset from HuggingFace
print("Loading dataset from HuggingFace...")
dataset = load_dataset("foodvisor-nyu/labeled-food-ingredients")

# Convert to pandas DataFrame
df = dataset["train"].to_pandas()

# Save to CSV
df.to_csv("data/ingredients.csv", index=False)
print(f"Dataset saved! Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample data:\n{df.head()}")