import cv2
import pytesseract
import numpy as np
from PIL import Image
import io
import os

class OCREngine:
    def __init__(self):
        # Configure Tesseract path
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Cache for frequently used configurations
        self.config_cache = {}
    
    def preprocess_image(self, image_array, preprocessing_level='auto'):
        """Enhanced preprocessing with multiple strategies"""
        try:
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_array
            
            # Quick check for image quality
            if preprocessing_level == 'auto':
                # Calculate image sharpness using Laplacian variance
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Determine preprocessing based on image quality
                if laplacian_var < 100:  # Blurry image
                    preprocessing_level = 'aggressive'
                elif laplacian_var > 500:  # Sharp image
                    preprocessing_level = 'minimal'
                else:
                    preprocessing_level = 'moderate'
            
            # Apply preprocessing based on level
            if preprocessing_level == 'minimal':
                # Just resize and denoise
                gray = self._resize_if_needed(gray)
                gray = cv2.fastNlMeansDenoising(gray, h=10)
                
            elif preprocessing_level == 'moderate':
                # Your current approach
                gray = self._resize_if_needed(gray)
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
                
            elif preprocessing_level == 'aggressive':
                # For difficult images
                gray = self._resize_if_needed(gray, min_height=1500)  # Larger upscale
                
                # Remove noise
                gray = cv2.medianBlur(gray, 3)
                
                # Sharpen image
                kernel = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
                gray = cv2.filter2D(gray, -1, kernel)
                
                # Morphological operations to enhance text
                kernel = np.ones((1, 1), np.uint8)
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
                
                # Thresholding
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, 3)
            
            return gray, preprocessing_level
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return image_array, 'error'
    
    def _resize_if_needed(self, gray, min_height=1000):
        """Resize image if too small"""
        height, width = gray.shape
        if height < min_height:
            scale_factor = min_height / height
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return gray
    
    def extract_text(self, image_path=None, image_bytes=None, psm_mode=6, 
                     language='eng', preprocessing_level='auto'):
        """Enhanced text extraction with multiple strategies"""
        try:
            # Load image
            if image_path:
                img = cv2.imread(image_path)
                if img is None:
                    return "Error: Could not read image file"
            elif image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                return "Error: No image provided"
            
            # Try multiple preprocessing strategies if first attempt fails
            strategies = [
                ('original', lambda x: x),
                ('grayscale', lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if len(x.shape) == 3 else x),
                ('threshold', self._simple_threshold),
                ('adaptive', lambda x: self._adaptive_threshold(x)),
            ]
            
            best_text = ""
            best_confidence = 0
            
            for strategy_name, strategy_func in strategies:
                try:
                    # Apply strategy
                    if strategy_name == 'original':
                        processed = img
                    else:
                        processed = strategy_func(img)
                    
                    # Get OCR data with confidence
                    custom_config = f'--psm {psm_mode} -l {language}'
                    data = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate confidence
                    confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        
                        # Extract text
                        text = ' '.join([data['text'][i] for i in range(len(data['text'])) 
                                        if data['text'][i].strip()])
                        
                        # Update best result
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_text = text
                            
                except Exception as e:
                    continue
            
            # Clean up text
            if best_text:
                lines = [line.strip() for line in best_text.split('\n') if line.strip()]
                best_text = ' '.join(lines)
                return f"{best_text}\n[Confidence: {best_confidence:.1f}%]"
            
            return "No text detected"
            
        except Exception as e:
            return f"OCR Error: {str(e)}"
    
    def _simple_threshold(self, img):
        """Simple binary threshold"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh
    
    def _adaptive_threshold(self, img):
        """Adaptive threshold"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    
    def extract_text_from_file(self, file):
        """Extract text from uploaded file object"""
        file_bytes = file.read()
        return self.extract_text(image_bytes=file_bytes)
    
    def extract_text_batch(self, image_paths, **kwargs):
        """Process multiple images efficiently"""
        results = {}
        for path in image_paths:
            results[path] = self.extract_text(image_path=path, **kwargs)
        return results

# Create global instance
ocr_engine = OCREngine()

if __name__ == "__main__":
    # Test with multiple strategies
    print("Testing Enhanced OCR Engine...")
    
    # Create test images with different qualities
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Create test images
    test_images = []
    
    # Clear image
    img1 = Image.new('RGB', (400, 200), color='white')
    d = ImageDraw.Draw(img1)
    d.text((10, 10), "Ingredients:", fill='black')
    d.text((10, 50), "Water, Sodium Benzoate,", fill='black')
    d.text((10, 90), "Natural Flavors, Aspartame", fill='black')
    img1.save("test_ingredients_clear.png")
    test_images.append("test_ingredients_clear.png")
    
    # Noisy image (simulate)
    img2 = np.array(img1)
    noise = np.random.normal(0, 25, img2.shape).astype(np.uint8)
    img2 = cv2.add(img2, noise)
    cv2.imwrite("test_ingredients_noisy.png", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    test_images.append("test_ingredients_noisy.png")
    
    # Test batch processing
    results = ocr_engine.extract_text_batch(test_images)
    
    for path, text in results.items():
        print(f"\nResults for {path}:")
        print(text)