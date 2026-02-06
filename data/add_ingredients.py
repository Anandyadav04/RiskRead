# add_safe_ingredients.py
import pandas as pd
import numpy as np

def add_safe_ingredients():
    df = pd.read_csv("data/ingredients.csv")
    
    # List of common safe ingredients with their labels (0 = Not Harmful)
    safe_ingredients = [
        ('cornflour', 0, 'Common thickening agent'),
        ('corn flour', 0, 'Common thickening agent'),
        ('onion', 0, 'Common vegetable'),
        ('onions', 0, 'Common vegetable'),
        ('lemon', 0, 'Common fruit'),
        ('lemons', 0, 'Common fruit'),
        ('flavor', 0, 'General flavoring'),
        ('flavors', 0, 'General flavoring'),
        ('flavour', 0, 'General flavoring'),
        ('flavours', 0, 'General flavoring'),
        ('color', 0, 'Food coloring'),
        ('colour', 0, 'Food coloring'),
        ('oil', 0, 'Cooking oil'),
        ('vegetable oil', 0, 'Cooking oil'),
        ('salt', 0, 'Essential mineral'),
        ('sugar', 0, 'Sweetener'),
        ('water', 0, 'Essential'),
        ('milk', 0, 'Dairy product'),
        ('egg', 0, 'Animal product'),
        ('eggs', 0, 'Animal product'),
        ('wheat flour', 0, 'Basic ingredient'),
        ('rice', 0, 'Staple food'),
        ('tomato', 0, 'Vegetable'),
        ('garlic', 0, 'Vegetable'),
        ('ginger', 0, 'Spice'),
        ('turmeric', 0, 'Spice'),
        ('pepper', 0, 'Spice'),
        ('vinegar', 0, 'Acidic liquid'),
        ('honey', 0, 'Natural sweetener'),
        ('yogurt', 0, 'Dairy product'),
        ('cheese', 0, 'Dairy product'),
        ('butter', 0, 'Dairy product'),
        ('cream', 0, 'Dairy product'),
        ('chocolate', 1, 'Generally safe but high sugar'),
        ('cocoa', 1, 'Generally safe in moderation'),
        ('vanilla', 0, 'Natural flavoring'),
        ('baking powder', 0, 'Leavening agent'),
        ('baking soda', 0, 'Leavening agent'),
        ('yeast', 0, 'Leavening agent'),
    ]
    
    # Check which ones are missing
    existing_ingredients = set(df['ingredient'].str.lower())
    new_rows = []
    
    for ing, label, note in safe_ingredients:
        if ing.lower() not in existing_ingredients:
            new_rows.append({
                'ingredient': ing,
                'int_label': label,
                'note': note
            })
            print(f"Adding: {ing} -> Label {label}")
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # If your CSV doesn't have 'note' column, add it
        if 'note' not in df.columns:
            df['note'] = ''
        
        # Combine
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Save improved dataset
        df.to_csv("data/ingredients_improved.csv", index=False)
        print(f"\n✅ Added {len(new_rows)} safe ingredients to dataset")
        print(f"✅ Total ingredients: {len(df)}")
    else:
        print("ℹ️ All safe ingredients already in dataset")
    
    return df

if __name__ == "__main__":
    df = add_safe_ingredients()
    print("\nDataset preview:")
    print(df[['ingredient', 'int_label']].tail(20))