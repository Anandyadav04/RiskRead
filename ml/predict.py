import pickle
import re

class IngredientClassifier:
    def __init__(self, model_path="ml/model.pkl"):
        # Try to load model
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.has_model = True
            print("✅ ML model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            self.model = None
            self.has_model = False
        
        # LABELS mapping
        self.LABELS = {
            0: ("Not Harmful", "✅ Generally safe for consumption"),
            1: ("Controversial", "⚠️ Mixed safety reviews - consume with caution"),
            2: ("Harmful", "🚫 Potential health risks - avoid or limit")
        }
        
        # EXTENSIVE SAFE OVERRIDES
        self.SAFE_OVERRIDES = {
            # === GRAINS & FLOURS ===
            'flour': (0, "✅ Basic cooking ingredient"),
            'wheat flour': (0, "✅ Basic baking ingredient"),
            'enriched wheat flour': (0, "✅ Fortified baking flour"),
            'cornflour': (0, "✅ Common thickening agent"),
            'corn flour': (0, "✅ Common thickening agent"),
            'cornstarch': (0, "✅ Common thickening agent"),
            'corn starch': (0, "✅ Common thickening agent"),
            'rice flour': (0, "✅ Gluten-free flour"),
            
            # === SWEETENERS ===
            'sugar': (0, "✅ Sweetener - safe in moderation"),
            'glucose': (0, "✅ Simple sugar, energy source"),
            'fructose': (0, "✅ Fruit sugar, natural sweetener"),
            'sugar/glucose-fructose': (0, "✅ Sweetener blend"),
            
            # === OILS & FATS ===
            'oil': (0, "✅ Cooking fat, essential in moderation"),
            'palm oil': (1, "⚠️ Environmental concerns but generally safe"),
            'palm kernel oil': (1, "⚠️ From palm kernels, similar to palm oil"),
            'canola oil': (0, "✅ Common cooking oil"),
            'soy oil': (0, "✅ Cooking oil from soybeans"),
            'canola and/or soy oil': (0, "✅ Blend of cooking oils"),
            
            # === ACIDS & PRESERVATIVES ===
            'citric acid': (0, "✅ Natural acid from citrus fruits"),
            'sodium citrate': (0, "✅ Salt of citric acid, preservative"),
            
            # === STARCHES & THICKENERS ===
            'tapioca dextrin': (0, "✅ Processed tapioca starch"),
            'modified corn starch': (1, "⚠️ Processed starch, generally safe"),
            'dextrin': (0, "✅ Soluble fiber from starch"),
            
            # === FLAVORS ===
            'natural flavors': (1, "⚠️ Source varies - generally safe"),
            'artificial flavors': (1, "⚠️ Synthetic - some concerns"),
            'natural and artificial flavors': (1, "⚠️ Blend of natural and synthetic flavors"),
            
            # === LEAVENING AGENTS ===
            'sodium bicarbonate': (0, "✅ Baking soda, common leavening agent"),
            'baking powder': (0, "✅ Common leavening agent"),
            'baking soda': (0, "✅ Common leavening agent"),
            
            # === BASIC INGREDIENTS ===
            'salt': (0, "✅ Essential mineral in moderation"),
            'water': (0, "✅ Essential for life"),
            'eggs': (0, "✅ Protein source, nutritious"),
            'whole eggs': (0, "✅ Complete protein source"),
            'milk': (0, "✅ Dairy product, calcium source"),
            'butter': (0, "✅ Dairy fat, natural"),
            'cheese': (0, "✅ Dairy product"),
            
            # === VEGETABLES & FRUITS ===
            'onion': (0, "✅ Common vegetable, nutritious"),
            'garlic': (0, "✅ Seasoning, has health benefits"),
            'tomato': (0, "✅ Vegetable/fruit, nutritious"),
            'lemon': (0, "✅ Citrus fruit, vitamin C source"),
            'apple': (0, "✅ Common fruit, nutritious"),
            'banana': (0, "✅ Fruit, potassium source"),
            
            # === PROTEINS ===
            'chicken': (0, "✅ Lean protein source"),
            'beef': (0, "✅ Protein and iron source"),
            'fish': (0, "✅ Lean protein, omega-3 source"),
            
            # === GRAINS ===
            'rice': (0, "✅ Staple grain"),
            'bread': (0, "✅ Baked food"),
            'pasta': (0, "✅ Wheat product"),
        }
        
        # HARMFUL OVERRIDES
        self.HARMFUL_OVERRIDES = {
            'hydrogenated': (2, "🚫 Contains trans fats, increases heart disease risk"),
            'hydrogenated palm kernel oil': (2, "🚫 Contains trans fats, unhealthy"),
            'trans fat': (2, "🚫 Increases heart disease risk"),
            'partially hydrogenated': (2, "🚫 Source of trans fats"),
            'aspartame': (2, "🚫 Artificial sweetener with health concerns"),
            'saccharin': (2, "🚫 Artificial sweetener, potential carcinogen"),
            'sodium benzoate': (2, "🚫 Preservative linked to hyperactivity"),
            'bha': (2, "🚫 Preservative, potential carcinogen"),
            'bht': (2, "🚫 Preservative, potential carcinogen"),
            'potassium bromate': (2, "🚫 Flour additive, banned in many countries"),
            'azodicarbonamide': (2, "🚫 Flour bleaching agent, industrial chemical"),
            'yellow 5': (2, "🚫 Artificial color, potential allergen"),
            'yellow 6': (2, "🚫 Artificial color, potential health risks"),
            'red 40': (2, "🚫 Artificial color, potential hyperactivity trigger"),
            'blue 1': (2, "🚫 Artificial color, potential health risks"),
            'blue 2': (2, "🚫 Artificial color, potential health risks"),
        }
        
        # CONTROVERSIAL OVERRIDES
        self.CONTROVERSIAL_OVERRIDES = {
            'high fructose corn syrup': (1, "⚠️ Linked to obesity and diabetes"),
            'corn syrup': (1, "⚠️ Sweetener with high fructose content"),
            'msg': (1, "⚠️ Monosodium glutamate - controversial"),
            'monosodium glutamate': (1, "⚠️ Flavor enhancer, controversial"),
            'artificial color': (1, "⚠️ Synthetic coloring, some concerns"),
            'artificial colors': (1, "⚠️ Synthetic colorings, some concerns"),
            'carrageenan': (1, "⚠️ Thickener with safety debates"),
            'xanthan gum': (1, "⚠️ Thickener, can cause digestive issues"),
            'soy lecithin': (1, "⚠️ Emulsifier, soy allergies common"),
        }
    
    def predict_ingredient(self, ingredient):
        """Predict safety of an ingredient"""
        ingredient_lower = ingredient.lower().strip()
        
        # 1. Check exact matches in overrides
        if ingredient_lower in self.SAFE_OVERRIDES:
            pred, explanation = self.SAFE_OVERRIDES[ingredient_lower]
            label, _ = self.LABELS[pred]
            return label, explanation
        
        if ingredient_lower in self.HARMFUL_OVERRIDES:
            pred, explanation = self.HARMFUL_OVERRIDES[ingredient_lower]
            label, _ = self.LABELS[pred]
            return label, explanation
        
        if ingredient_lower in self.CONTROVERSIAL_OVERRIDES:
            pred, explanation = self.CONTROVERSIAL_OVERRIDES[ingredient_lower]
            label, _ = self.LABELS[pred]
            return label, explanation
        
        # 2. Check for harmful keywords
        harmful_keywords = ['hydrogenated', 'aspartame', 'saccharin', 'bha', 'bht', 
                          'yellow 5', 'yellow 6', 'red 40', 'blue 1', 'blue 2',
                          'artificial color', 'artificial colours']
        
        for keyword in harmful_keywords:
            if keyword in ingredient_lower:
                return "Harmful", f"🚫 Contains {keyword} - potential health risk"
        
        # 3. Check for controversial keywords
        controversial_keywords = ['corn syrup', 'msg', 'monosodium glutamate', 
                                'artificial flavor', 'high fructose']
        
        for keyword in controversial_keywords:
            if keyword in ingredient_lower:
                return "Controversial", f"⚠️ Contains {keyword} - mixed safety reviews"
        
        # 4. Check for safe patterns
        safe_patterns = [
            ('flour', "✅ Common food ingredient"),
            ('salt', "✅ Essential mineral"),
            ('sugar', "✅ Sweetener in moderation"),
            ('oil', "✅ Cooking fat"),
            ('water', "✅ Essential for life"),
            ('milk', "✅ Dairy product"),
            ('egg', "✅ Protein source"),
            ('rice', "✅ Staple grain"),
            ('bread', "✅ Baked food"),
            ('cheese', "✅ Dairy product"),
            ('acid', "✅ Common food acid"),
            ('starch', "✅ Thickening agent"),
            ('dextrin', "✅ Soluble fiber"),
            ('citrate', "✅ Preservative"),
        ]
        
        for pattern, explanation in safe_patterns:
            if pattern in ingredient_lower:
                return "Not Harmful", explanation
        
        # 5. Try ML model if available
        if self.has_model:
            try:
                pred = self.model.predict([ingredient])[0]
                label, explanation = self.LABELS[pred]
                return label, explanation
            except:
                pass
        
        # 6. Default: safe
        return "Not Harmful", "✅ Assuming safe unless known to be harmful"
    
    def predict_multiple(self, ingredients):
        """Predict safety for multiple ingredients"""
        results = []
        for ing in ingredients:
            if ing and ing.strip():
                label, explanation = self.predict_ingredient(ing.strip())
                results.append({
                    'ingredient': ing.strip(),
                    'label': label,
                    'explanation': explanation
                })
        return results

# Create a global instance
classifier = IngredientClassifier()

# Quick test
if __name__ == "__main__":
    print("Testing classifier...")
    test_ingredients = [
        'enriched wheat flour',
        'sugar/glucose-fructose',
        'hydrogenated palm kernel oil',
        'citric acid',
        'modified corn starch',
        'natural and artificial flavors',
        'sodium citrate',
        'yellow 5 lake',
        'salt',
        'baking powder'
    ]
    
    for ing in test_ingredients:
        label, explanation = classifier.predict_ingredient(ing)
        print(f"{ing:35} -> {label:15} ({explanation[:40]}...)")