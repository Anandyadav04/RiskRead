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
        """Improved preprocessing for better OCR results"""
        try:
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_array
            
            # Resize if too small
            height, width = gray.shape
            if height < 100 or width < 100:
                scale = max(300/height, 300/width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Apply adaptive thresholding (better for uneven lighting)
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            
            # Remove noise
            gray = cv2.medianBlur(gray, 3)
            
            # Enhance contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=50)
            
            return gray
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return image_array
    
    def extract_text(self, image_path=None, image_bytes=None):
        """Extract text from image file or bytes"""
        try:
            if image_path:
                # Read from file path
                img = cv2.imread(image_path)
                if img is None:
                    return "Error: Could not read image file"
            elif image_bytes:
                # Read from bytes
                image = Image.open(io.BytesIO(image_bytes))
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                return "Error: No image provided"
            
            # Preprocess
            processed = self.preprocess_image(img)
            
            # Extract text
            text = pytesseract.image_to_string(processed)
            
            # Clean up text
            text = ' '.join(text.split())  # Remove extra whitespace
            
            return text if text.strip() else "No text detected"
            
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