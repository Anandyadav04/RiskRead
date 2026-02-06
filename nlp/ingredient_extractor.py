import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class IngredientExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Common ingredient separators
        self.separators = [',', ';', '\n', '•', ' and ', ' & ', ' or ']
        
        # Common measurement words to remove
        self.measurements = [
            'tsp', 'tbsp', 'cup', 'cups', 'oz', 'ounce', 'ounces',
            'lb', 'pound', 'pounds', 'g', 'gram', 'grams', 'mg',
            'milligram', 'milligrams', 'ml', 'milliliter', 'milliliters',
            'liter', 'liters', 'package', 'packages', 'can', 'cans',
            'mg', 'g', 'kg', 'ml', 'l', 'dl'
        ]
        
        # Common prefixes to remove
        self.prefixes = [
            'contains', 'ingredients', 'ingredient', 'made with',
            'made of', 'composed of', 'consists of', 'including'
        ]
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common prefixes with colon
        for prefix in self.prefixes:
            text = re.sub(rf'{prefix}[:\s]*', '', text, flags=re.IGNORECASE)
        
        # Remove "less than X% of" pattern
        text = re.sub(r'less than\s*\d+%\s*of[:\s]*', '', text, flags=re.IGNORECASE)
        
        # Remove square brackets
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # Remove common measurement patterns
        for unit in self.measurements:
            # Match patterns like "10g", "10 g", "10 grams"
            text = re.sub(rf'\d+\s*{unit}s?\b', ' ', text, flags=re.IGNORECASE)
        
        # Remove percentage signs and numbers with them
        text = re.sub(r'\d+%', ' ', text)
        
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', ' ', text)
        
        # Remove special characters but keep commas, spaces, letters, and slashes
        text = re.sub(r'[^\w\s,/&+()-]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def extract_ingredients(self, text):
        """Extract individual ingredients from text - SIMPLE VERSION"""
        if not text or not text.strip():
            return []
        
        print(f"NLP DEBUG: Original text: {text[:200]}...")
        
        # Clean text
        cleaned_text = self.clean_text(text)
        print(f"NLP DEBUG: Cleaned text: {cleaned_text[:200]}...")
        
        # Replace all separators with commas
        for sep in [';', '\n', '•', '/', ' and ', ' & ', ' or ']:
            cleaned_text = cleaned_text.replace(sep, ',')
        
        # Split by commas
        parts = [p.strip() for p in cleaned_text.split(',')]
        
        # Process each part
        ingredients = []
        for part in parts:
            if not part or len(part) < 2:
                continue
            
            # Remove trailing/leading special chars
            part = re.sub(r'^[\s,\-\.]+|[\s,\-\.]+$', '', part)
            
            # Skip measurement percentages
            if re.match(r'^\d+%$', part):
                continue
            
            # Skip if it's just a filler word
            filler_words = ['a', 'an', 'the', 'of', 'with', 'and', 'or', 'but']
            if part.lower() in filler_words:
                continue
            
            # Skip if too short
            if len(part) < 2:
                continue
            
            ingredients.append(part)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ingredients = []
        for ing in ingredients:
            ing_lower = ing.lower()
            if ing_lower not in seen and len(ing) > 2:
                seen.add(ing_lower)
                unique_ingredients.append(ing)
        
        print(f"NLP DEBUG: Extracted ingredients: {unique_ingredients}")
        return unique_ingredients
    
    def extract_from_ocr(self, ocr_text):
        """Specialized extraction for OCR text - SIMPLE RELIABLE VERSION"""
        if not ocr_text or ocr_text.strip() == "No text detected":
            print("NLP DEBUG: OCR returned no text")
            return []
        
        print(f"NLP DEBUG: OCR text received: {ocr_text[:200]}...")
        
        # Convert to lowercase
        text = ocr_text.lower()
        
        # Remove everything before first colon (like "MADE OF:")
        if ':' in text:
            parts = text.split(':', 1)
            if len(parts) > 1:
                text = parts[1]
        
        # Remove "LESS THAN X% OF" pattern
        text = re.sub(r'less than\s*\d+%\s*of[:\s]*', '', text, flags=re.IGNORECASE)
        
        # Fix common OCR errors
        corrections = {
            'gorn': 'corn',
            'tap ioga': 'tapioca',
            'st argh': 'starch',
            'artif igial': 'artificial',
            'nat ural': 'natural',
        }
        
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        
        print(f"NLP DEBUG: Processed text: {text[:200]}...")
        
        return self.extract_ingredients(text)

# Create a global instance
ingredient_extractor = IngredientExtractor()

if __name__ == "__main__":
    # Test with the Skittles OCR text
    test_text = """MADE OF: SUGAR, GORN SYRUP, HYDROGENATED PALM KERNEL OIL; LESS THAN 2% OF: CITRIC ACID, TAPIOGA DEXTRIN, MODIFIED CORN STARGH, NATURAL AND ARTIFIGIAL FLAVORS, SODIUM CITRATE, COLORS (YELLOW 5 LAKE, RED 40, BLUE 1 LAKE, YELLOW 6 LAKE, BLUE 2 LAKE)"""
    
    print("Testing Ingredient Extractor")
    print("=" * 60)
    
    ingredients = ingredient_extractor.extract_from_ocr(test_text)
    
    print(f"\n✅ Extracted {len(ingredients)} ingredients:")
    for i, ing in enumerate(ingredients, 1):
        print(f"{i:2}. {ing}")