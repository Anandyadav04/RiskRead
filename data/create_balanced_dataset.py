import pandas as pd
import numpy as np
import random

def create_balanced_dataset():
    # Load original dataset
    try:
        df = pd.read_csv("data/ingredients.csv")
        print(f"Loaded original dataset: {len(df)} rows")
    except:
        print("Creating new dataset from scratch")
        df = pd.DataFrame(columns=['ingredient', 'int_label'])
    
    # Add LOTS of safe ingredients (Label 0)
    safe_ingredients = [
        ('cornflour', 0),
        ('corn flour', 0),
        ('wheat flour', 0),
        ('all purpose flour', 0),
        ('bread flour', 0),
        ('rice flour', 0),
        ('salt', 0),
        ('sea salt', 0),
        ('onion', 0),
        ('onions', 0),
        ('garlic', 0),
        ('tomato', 0),
        ('tomatoes', 0),
        ('carrot', 0),
        ('carrots', 0),
        ('potato', 0),
        ('potatoes', 0),
        ('spinach', 0),
        ('lettuce', 0),
        ('cabbage', 0),
        ('broccoli', 0),
        ('cauliflower', 0),
        ('cucumber', 0),
        ('bell pepper', 0),
        ('mushroom', 0),
        ('apple', 0),
        ('banana', 0),
        ('orange', 0),
        ('lemon', 0),
        ('lime', 0),
        ('grape', 0),
        ('strawberry', 0),
        ('blueberry', 0),
        ('mango', 0),
        ('pineapple', 0),
        ('sugar', 0),
        ('brown sugar', 0),
        ('cane sugar', 0),
        ('water', 0),
        ('filtered water', 0),
        ('oil', 0),
        ('vegetable oil', 0),
        ('olive oil', 0),
        ('coconut oil', 0),
        ('canola oil', 0),
        ('sunflower oil', 0),
        ('butter', 0),
        ('ghee', 0),
        ('milk', 0),
        ('whole milk', 0),
        ('skim milk', 0),
        ('cream', 0),
        ('yogurt', 0),
        ('cheese', 0),
        ('cheddar cheese', 0),
        ('mozzarella cheese', 0),
        ('parmesan cheese', 0),
        ('egg', 0),
        ('eggs', 0),
        ('whole eggs', 0),
        ('egg white', 0),
        ('egg yolk', 0),
        ('chicken', 0),
        ('beef', 0),
        ('pork', 0),
        ('fish', 0),
        ('salmon', 0),
        ('tuna', 0),
        ('shrimp', 0),
        ('tofu', 0),
        ('lentils', 0),
        ('beans', 0),
        ('chickpeas', 0),
        ('kidney beans', 0),
        ('black pepper', 0),
        ('white pepper', 0),
        ('turmeric', 0),
        ('cumin', 0),
        ('coriander', 0),
        ('cinnamon', 0),
        ('nutmeg', 0),
        ('ginger', 0),
        ('basil', 0),
        ('oregano', 0),
        ('thyme', 0),
        ('rosemary', 0),
        ('parsley', 0),
        ('cilantro', 0),
        ('mint', 0),
        ('dill', 0),
        ('vanilla', 0),
        ('vanilla extract', 0),
        ('honey', 0),
        ('maple syrup', 0),
        ('agave syrup', 0),
        ('molasses', 0),
        ('baking powder', 0),
        ('baking soda', 0),
        ('sodium bicarbonate', 0),
        ('yeast', 0),
        ('vinegar', 0),
        ('white vinegar', 0),
        ('apple cider vinegar', 0),
        ('balsamic vinegar', 0),
        ('mustard', 0),
        ('ketchup', 0),
        ('mayonnaise', 0),
        ('soy sauce', 0),
        ('rice', 0),
        ('white rice', 0),
        ('brown rice', 0),
        ('oats', 0),
        ('oatmeal', 0),
        ('quinoa', 0),
        ('barley', 0),
        ('corn', 0),
        ('pasta', 0),
        ('noodles', 0),
        ('bread', 0),
        ('whole wheat bread', 0),
        ('almond', 0),
        ('walnut', 0),
        ('peanut', 0),
        ('cashew', 0),
        ('sunflower seed', 0),
        ('pumpkin seed', 0),
        ('chia seed', 0),
        ('flaxseed', 0),
        ('sesame seed', 0),
        ('tea', 0),
        ('green tea', 0),
        ('black tea', 0),
        ('coffee', 0),
        ('juice', 0),
        ('apple juice', 0),
        ('orange juice', 0),
        ('gelatin', 0),
        ('pectin', 0),
        ('breadcrumbs', 0),
        ('stock', 0),
        ('broth', 0),
        ('bouillon', 0),
    ]
    
    # Add controversial ingredients (Label 1)
    controversial_ingredients = [
        ('high fructose corn syrup', 1),
        ('corn syrup', 1),
        ('palm oil', 1),
        ('carrageenan', 1),
        ('soy lecithin', 1),
        ('xanthan gum', 1),
        ('monosodium glutamate', 1),
        ('msg', 1),
        ('sulfites', 1),
        ('guar gum', 1),
        ('cellulose gum', 1),
        ('maltodextrin', 1),
        ('dextrose', 1),
        ('invert sugar', 1),
        ('corn syrup solids', 1),
        ('modified food starch', 1),
        ('hydrogenated oil', 1),
        ('propylene glycol', 1),
        ('sorbic acid', 1),
        ('benzoic acid', 1),
        ('artificial flavor', 1),
        ('artificial flavours', 1),
        ('natural flavor', 1),
        ('natural flavours', 1),
        ('food color', 1),
        ('food colour', 1),
        ('natural color', 1),
        ('natural colour', 1),
        ('chocolate', 1),
        ('dark chocolate', 1),
        ('milk chocolate', 1),
        ('white chocolate', 1),
    ]
    
    # Add harmful ingredients (Label 2)
    harmful_ingredients = [
        ('trans fat', 2),
        ('partially hydrogenated oil', 2),
        ('aspartame', 2),
        ('saccharin', 2),
        ('sodium benzoate', 2),
        ('bha', 2),
        ('bht', 2),
        ('potassium bromate', 2),
        ('azodicarbonamide', 2),
        ('olestra', 2),
        ('sodium nitrite', 2),
        ('sodium nitrate', 2),
        ('potassium sorbate', 2),
        ('sodium bisulfite', 2),
        ('butylated hydroxyanisole', 2),
        ('butylated hydroxytoluene', 2),
        ('propyl gallate', 2),
        ('tertiary butylhydroquinone', 2),
        ('artificial colors', 2),
        ('red 40', 2),
        ('yellow 5', 2),
        ('yellow 6', 2),
        ('blue 1', 2),
        ('blue 2', 2),
        ('artificial sweetener', 2),
        ('synthetic flavor', 2),
        ('processed oil', 2),
    ]
    
    # Create new balanced dataset
    all_ingredients = safe_ingredients + controversial_ingredients + harmful_ingredients
    
    # Add variations for each ingredient
    enhanced_ingredients = []
    for ing, label in all_ingredients:
        enhanced_ingredients.append((ing, label))
        
        # Add some variations
        if ' ' in ing:
            # Add without spaces
            enhanced_ingredients.append((ing.replace(' ', ''), label))
        
        # Add capitalized version
        enhanced_ingredients.append((ing.title(), label))
    
    # Create DataFrame
    balanced_df = pd.DataFrame(enhanced_ingredients, columns=['ingredient', 'int_label'])
    
    # Remove duplicates
    balanced_df = balanced_df.drop_duplicates(subset=['ingredient'])
    
    # Add some random safe combinations
    safe_words = ['fresh', 'organic', 'natural', 'pure', 'raw', 'whole', 'dried']
    base_safe = ['flour', 'salt', 'sugar', 'oil', 'water', 'milk', 'eggs']
    
    for i in range(50):
        word1 = random.choice(safe_words)
        word2 = random.choice(base_safe)
        ingredient = f"{word1} {word2}"
        if ingredient not in balanced_df['ingredient'].values:
            balanced_df = pd.concat([balanced_df, pd.DataFrame([{
                'ingredient': ingredient,
                'int_label': 0
            }])], ignore_index=True)
    
    print(f"Created balanced dataset with {len(balanced_df)} ingredients")
    print(f"Class distribution:")
    for label in [0, 1, 2]:
        count = (balanced_df['int_label'] == label).sum()
        percentage = (count / len(balanced_df)) * 100
        label_name = ['Not Harmful', 'Controversial', 'Harmful'][label]
        print(f"  {label_name}: {count} ({percentage:.1f}%)")
    
    # Save
    balanced_df.to_csv("data/balanced_ingredients.csv", index=False)
    print("\n✅ Balanced dataset saved as 'data/balanced_ingredients.csv'")
    
    return balanced_df

if __name__ == "__main__":
    create_balanced_dataset()