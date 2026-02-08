from flask import Flask, render_template, request, jsonify, flash
import os
import sys
import base64
from PIL import Image
import io
from werkzeug.utils import secure_filename

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.ocr_engine import ocr_engine
from nlp.ingredient_extractor import ingredient_extractor
from nlp.post_processor import post_processor
from ml.predict import classifier

app = Flask(__name__)
app.secret_key = 'riskread-secret-key-2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_base64_image(base64_data, filename="pasted_image.png"):
    """Convert base64 string to image file"""
    try:
        # Decode base64
        image_data = base64.b64decode(base64_data)
        
        # Create image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Save to uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        
        print(f"✅ Saved base64 image to: {filepath}")
        print(f"   Image size: {image.size}, Mode: {image.mode}")
        
        return filepath
    except Exception as e:
        print(f"❌ Error saving base64 image: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze ingredients from text, file upload, or pasted image"""
    print("\n" + "="*60)
    print("DEBUG: /analyze endpoint called")
    print("="*60)
    
    # DEBUG: Print all form data
    print("📋 Form data received:")
    for key in request.form:
        if key == 'image_data':
            data_len = len(request.form[key]) if request.form[key] else 0
            print(f"  {key}: [base64 data, length: {data_len}]")
        else:
            print(f"  {key}: {request.form[key][:100] if request.form[key] else 'None'}")
    
    # DEBUG: Print files
    print("📁 Files received:")
    for key in request.files:
        file = request.files[key]
        print(f"  {key}: {file.filename if file.filename else 'No filename'}")
    
    try:
        results = []
        source_type = "text"
        extracted_text = ""
        filepath = None
        
        # ===== CHECK 1: PASTED IMAGE (base64 data) =====
        image_data = request.form.get('image_data')
        if image_data and image_data.strip() and len(image_data.strip()) > 100:
            print("📋 DEBUG: Processing PASTED IMAGE (base64)")
            print(f"  Base64 data length: {len(image_data)}")
            source_type = "image"
            
            # Save base64 image to file
            filepath = save_base64_image(image_data, "pasted_image.png")
            
            if filepath and os.path.exists(filepath):
                print(f"  ✅ Saved pasted image to: {filepath}")
                
                # Extract text using OCR
                print(f"  Starting OCR extraction...")
                try:
                    extracted_text = ocr_engine.extract_text(filepath)
                    print(f"  OCR Result: {extracted_text[:200]}...")
                    
                    if not extracted_text or not extracted_text.strip() or "No text detected" in extracted_text:
                        print("  ⚠️ OCR returned empty or no text!")
                        # Don't flash error yet, try other sources
                    else:
                        print("  ✅ OCR successful")
                        
                except Exception as ocr_error:
                    print(f"  ❌ OCR ERROR: {str(ocr_error)}")
                    extracted_text = "OCR failed"
            else:
                print(f"  ❌ ERROR: Could not save pasted image")
                # Don't flash error yet, try other sources
        
        # ===== CHECK 2: UPLOADED IMAGE FILE =====
        elif 'image' in request.files:
            file = request.files['image']
            print(f"📁 DEBUG: Processing FILE UPLOAD")
            print(f"  Filename: {file.filename}")
            
            if file and file.filename != '':
                if allowed_file(file.filename):
                    source_type = "image"
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    
                    print(f"  Saving to: {filepath}")
                    file.save(filepath)
                    
                    if os.path.exists(filepath):
                        print(f"  ✅ File saved successfully")
                        
                        # Extract text using OCR
                        print(f"  Starting OCR extraction...")
                        try:
                            extracted_text = ocr_engine.extract_text(filepath)
                            print(f"  OCR Result: {extracted_text[:200]}...")
                            
                            if not extracted_text or not extracted_text.strip() or "No text detected" in extracted_text:
                                print("  ⚠️ OCR returned empty or no text!")
                            else:
                                print("  ✅ OCR successful")
                                
                        except Exception as ocr_error:
                            print(f"  ❌ OCR ERROR: {str(ocr_error)}")
                            extracted_text = "OCR failed"
                    else:
                        print(f"  ❌ ERROR: File was not saved!")
                else:
                    print(f"  ⚠️ File type NOT allowed: {file.filename}")
            else:
                print(f"  ℹ️ No valid file uploaded")
        
        # ===== CHECK 3: TEXT INPUT =====
        print(f"\n📝 DEBUG: Checking TEXT INPUT")
        text_input = request.form.get('ingredients', '')
        print(f"  Text input received: {text_input[:100]}...")
        
        if text_input and text_input.strip():
            # If we have text input, use it (overrides OCR if present)
            extracted_text = text_input
            source_type = "text"
            print(f"  ✅ Using text input")
        elif extracted_text and extracted_text.strip() and extracted_text != "OCR failed":
            # Use OCR result if available
            print(f"  ✅ Using OCR result")
        else:
            # No input at all
            print(f"  ❌ No input provided")
            flash("❌ Please enter ingredients, upload an image, or paste an image for analysis.", "error")
            return render_template('index.html')
        
        print(f"\n📊 DEBUG: Source type: {source_type}")
        print(f"DEBUG: Text to process: {extracted_text[:200]}...")
        
        # ===== PROCESS INGREDIENTS =====
        # Extract ingredients from text
        if source_type == "image":
            ingredients = ingredient_extractor.extract_from_ocr(extracted_text)
            print(f"  Extracted {len(ingredients)} ingredients from {source_type}")
        else:
            ingredients = ingredient_extractor.extract_ingredients(extracted_text)
            print(f"  Extracted {len(ingredients)} ingredients from text")
        
        print(f"  Raw ingredients: {ingredients}")
        
        # Apply post-processing
        if ingredients:
            try:
                ingredients = post_processor.clean_ingredient_list(ingredients)
                print(f"  Cleaned ingredients: {ingredients}")
            except Exception as e:
                print(f"  ⚠️ Post-processor failed: {e}, using raw ingredients")
        
        if not ingredients:
            print("  ⚠️ WARNING: No ingredients extracted!")
            
            # Try emergency extraction
            print("  Trying emergency extraction...")
            emergency_ingredients = []
            # Simple split by comma
            for part in extracted_text.split(','):
                part = part.strip()
                if part and len(part) > 3:
                    # Remove common prefixes
                    for prefix in ['made of:', 'contains:', 'ingredients:', 'less than']:
                        if part.lower().startswith(prefix):
                            part = part[len(prefix):].strip()
                    if part:
                        emergency_ingredients.append(part)
            
            if emergency_ingredients:
                ingredients = emergency_ingredients
                print(f"  ✅ Emergency extraction found: {ingredients}")
            else:
                flash("❌ No ingredients found. Please try again with clearer input.", "error")
                return render_template('index.html')
        
        # ===== MAKE PREDICTIONS =====
        print(f"\n🤖 DEBUG: Making predictions...")
        predictions = classifier.predict_multiple(ingredients)
        print(f"  Made {len(predictions)} predictions")
        
        # Calculate statistics
        stats = {
            'total': len(predictions),
            'harmful': sum(1 for p in predictions if p['label'] == 'Harmful'),
            'controversial': sum(1 for p in predictions if p['label'] == 'Controversial'),
            'safe': sum(1 for p in predictions if p['label'] == 'Not Harmful')
        }
        
        print(f"  Stats: {stats}")
        
        # Clean up temporary file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"  🗑️ Cleaned up temporary file: {filepath}")
            except:
                print(f"  ⚠️ Could not remove temporary file: {filepath}")
        
        print(f"\n✅ DEBUG: Rendering results template")
        print("="*60 + "\n")
        
        return render_template('result.html', 
                             predictions=predictions,
                             stats=stats,
                             source_type=source_type,
                             original_text=extracted_text[:500])
        
    except Exception as e:
        print(f"\n❌ ERROR in /analyze: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f"❌ An error occurred: {str(e)}", "error")
        return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    data = request.json
    ingredient = data.get('ingredient', '')
    
    if not ingredient:
        return jsonify({'error': 'No ingredient provided'}), 400
    
    label, explanation = classifier.predict_ingredient(ingredient)
    
    return jsonify({
        'ingredient': ingredient,
        'label': label,
        'explanation': explanation
    })

@app.errorhandler(413)
def too_large(e):
    return "File is too large. Maximum size is 32MB.", 413

if __name__ == '__main__':
    print("\n🌐 Open http://localhost:5000 in your browser")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)