import cv2
import pytesseract
import numpy as np
from PIL import Image
import io
import os

class OCREngine:
    def __init__(self):
        # Configure Tesseract path (update for your system)
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def preprocess_image(self, image_array):
        """Standard preprocessing: Upscale + Gentle Threshold"""
        try:
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_array
            
            # Upscale if needed (crucial for small text)
            height, width = gray.shape
            scale_factor = 2.0
            if height < 1000:
                scale_factor = max(2.0, 1000/height)
                
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Simple thresholding (works best for clear text)
            # Try a very simple binary threshold first if image quality is decent
            # But adaptive handles shadows. Let's use a very GENTLE adaptive.
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
                                        
            return gray
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return image_array
    
    def extract_text(self, image_path=None, image_bytes=None):
        """Extract text from image file or bytes"""
        try:
            if image_path:
                img = cv2.imread(image_path)
            elif image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                return "Error: No image provided"
            
            if img is None: return "Error: Could not process image"
            
            processed = self.preprocess_image(img)
            
            # Config: Use PSM 6 (Assume a single uniform block of text)
            # This is generally the most reliable for copy-pasted text blocks
            custom_config = r'--psm 6' 
            
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            # Post-Processing cleanup
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = ' '.join(lines)
            
            return text if text else "No text detected"
            
        except Exception as e:
            return f"OCR Error: {str(e)}"
    
    def extract_text_from_file(self, file):
        """Extract text from uploaded file object"""
        file_bytes = file.read()
        return self.extract_text(image_bytes=file_bytes)

# Create a global instance
ocr_engine = OCREngine()

if __name__ == "__main__":
    # Test with a sample image (create a test image first)
    print("Testing OCR Engine...")
    
    # Create a simple test image with text
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Create test image
    img = Image.new('RGB', (400, 200), color='white')
    d = ImageDraw.Draw(img)
    
    # Try to use a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    d.text((10, 10), "Ingredients:", fill='black', font=font)
    d.text((10, 50), "Water, Sodium Benzoate,", fill='black', font=font)
    d.text((10, 90), "Natural Flavors, Aspartame", fill='black', font=font)
    
    # Save test image
    img.save("test_ingredients.png")
    
    # Test OCR
    text = ocr_engine.extract_text("test_ingredients.png")
    print(f"Extracted text:\n{text}")