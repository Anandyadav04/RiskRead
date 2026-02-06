import re

class PostProcessor:
    def __init__(self):
        self.common_ingredients = [
            'water', 'salt', 'sugar', 'flour', 'eggs', 'oil', 'butter',
            'milk', 'chocolate', 'cocoa', 'vanilla', 'baking powder',
            'baking soda', 'yeast', 'honey', 'syrup', 'vinegar', 'spices'
        ]
        
        self.allergy_keywords = [
            'allergy', 'warning', 'contains', 'may contain', 'product',
            'information', 'packaged', 'manufactured', 'facility',
            'nuts', 'peanuts', 'tree nuts', 'allergen'
        ]
        
        self.ocr_corrections = {
            'ou ': 'whole ',
            'oya ': 'soy ',
            'ch ': '',
            'oo ': '',
            'q ': '',
            'artifical': 'artificial',
            'flavour': 'flavor',
            'flavours': 'flavors',
            '~~': '',
            '~~ ': '',
        }
    
    def clean_ingredient_list(self, ingredients):
        """Clean up extracted ingredients"""
        cleaned = []
        
        for ing in ingredients:
            original = ing
            
            # Remove if it's clearly not an ingredient
            if self._is_not_ingredient(ing):
                continue
            
            # Fix common OCR errors
            ing = self._fix_common_errors(ing)
            
            # Remove allergy information
            ing = self._remove_allergy_info(ing)
            
            # Skip if too short after cleaning
            if len(ing.strip()) < 2:
                continue
            
            # Split compound ingredients intelligently
            split_ingredients = self._split_compound(ing)
            
            for split_ing in split_ingredients:
                if split_ing and len(split_ing.strip()) > 2:
                    cleaned.append(split_ing.strip())
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for ing in cleaned:
            ing_lower = ing.lower()
            if ing_lower not in seen and len(ing) > 2:
                seen.add(ing_lower)
                unique.append(ing)
        
        return unique
    
    def _is_not_ingredient(self, text):
        """Check if text is clearly not an ingredient"""
        text_lower = text.lower()
        
        # Too short
        if len(text) < 2:
            return True
        
        # Contains allergy warnings
        if any(keyword in text_lower for keyword in self.allergy_keywords):
            return True
        
        # Mostly numbers or special chars
        if re.match(r'^[\d\W_]+$', text):
            return True
        
        # Looks like a page number or code
        if re.match(r'^[pq]\d+$', text_lower):
            return True
        
        return False
    
    def _fix_common_errors(self, text):
        """Fix common OCR/text errors"""
        for wrong, right in self.ocr_corrections.items():
            text = text.replace(wrong, right)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _remove_allergy_info(self, text):
        """Remove allergy warning text"""
        text_lower = text.lower()
        
        # Find position of first allergy keyword
        positions = []
        for keyword in self.allergy_keywords:
            pos = text_lower.find(keyword)
            if pos != -1:
                positions.append(pos)
        
        if positions:
            # Cut off at first allergy keyword
            cut_pos = min(positions)
            text = text[:cut_pos].strip()
        
        return text
    
    def _split_compound(self, text):
        """Split compound ingredient descriptions"""
        text_lower = text.lower()
        
        # Known compounds to keep together
        compounds = [
            'dark chocolate', 'milk chocolate', 'white chocolate',
            'baking powder', 'baking soda', 'whole eggs', 'egg whites',
            'egg yolks', 'brown sugar', 'cane sugar', 'palm oil',
            'olive oil', 'coconut oil', 'soybean oil', 'canola oil',
            'wheat flour', 'all-purpose flour', 'bread flour',
            'cake flour', 'cocoa powder', 'vanilla extract',
            'natural flavor', 'artificial flavor', 'chocolate chips',
            'chocolate chunk', 'modified palm', 'soya oil',
            'sodium bicarbonate', 'glucose-fructose'
        ]
        
        # Check if it's a known compound
        for compound in compounds:
            if compound in text_lower:
                return [text]  # Keep as is
        
        # Special case: "dark chocolate chunk chocolate chips"
        if 'dark chocolate chunk' in text_lower and 'chocolate chips' in text_lower:
            return ['dark chocolate chunk', 'chocolate chips']
        
        # Check if it contains "and" or "or" between ingredients
        if ' and ' in text_lower or ' or ' in text_lower:
            parts = re.split(r'\s+and\s+|\s+or\s+', text, flags=re.IGNORECASE)
            if len(parts) > 1:
                return [p.strip() for p in parts if p.strip()]
        
        # Check if it has slashes but no spaces (like sugar/glucose-fructose)
        if '/' in text and ' ' not in text:
            # Single compound with slash, keep as is
            return [text]
        
        return [text]

# Global instance
post_processor = PostProcessor()