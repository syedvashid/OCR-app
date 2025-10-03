from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import time
import os
from dotenv import load_dotenv 
# from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
from configration import llm1
from Database import save_user_correction, get_learned_patterns, apply_learned_corrections, create_image_hash, initialize_database, db

# TrOCR imports
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# TrOCR Configuration
class TrOCRConfig:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = os.getenv('TROCR_MODEL', 'microsoft/trocr-base-handwritten')  # Default to handwritten text
        self.processor = None
        self.model = None
        self.initialized = False
    
    def initialize_model(self):
        """Initialize TrOCR model and processor"""
        try:
            print(f"Initializing TrOCR model: {self.model_name}")
            print(f"Using device: {self.device}")
            
            if torch.cuda.is_available():
                print(f"CUDA available - GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print("Using CPU for processing")
            
            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Move to appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Enable memory optimization for local development
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear cache
                # Optional: Enable half precision for memory efficiency (uncomment if needed)
                # self.model = self.model.half()
            
            self.initialized = True
            print("✅ TrOCR model initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize TrOCR model: {e}")
            print("💡 Make sure you have installed: pip install torch transformers pillow")
            return False
    
    def is_ready(self):
        return self.initialized and self.processor is not None and self.model is not None

# Global TrOCR instance
trocr_config = TrOCRConfig()

def preprocess_image(image):
    """
    Preprocess image for better OCR results
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (TrOCR works better with reasonable sizes)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Optional: Enhance contrast (uncomment if needed)
        # from PIL import ImageEnhance
        # enhancer = ImageEnhance.Contrast(image)
        # image = enhancer.enhance(1.2)
        
        return image
        
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return image

import fitz  # PyMuPDF

import fitz  # PyMuPDF

def process_pdf_with_trocr(pdf_data):
    """
    Process PDF file using PyMuPDF (no poppler needed)
    """
    try:
        all_text = []
        
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        
        print(f"Processing PDF with {len(pdf_document)} page(s)")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            print(f"Processing page {page_num + 1}...")
            
            # Render page to image (300 DPI)
            mat = fitz.Matrix(300/72, 300/72)  # 300 DPI scaling
            pix = page.get_pixmap(matrix=mat)
            
            # Convert pixmap to PIL Image first, then to bytes
            # This fixes the "cannot identify image file" error
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert PIL Image to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            # Process with TrOCR
            page_text = process_with_trocr(img_data)
            
            if page_text.strip():
                all_text.append(f"[Page {page_num + 1}]\n{page_text}")
        
        pdf_document.close()
        
        # Combine all pages
        combined_text = "\n\n".join(all_text)
        return combined_text if combined_text else "No text detected in PDF"
        
    except Exception as error:
        print(f"PDF processing error: {error}")
        import traceback
        traceback.print_exc()
        raise Exception(f"PDF processing failed: {str(error)}")


def process_with_trocr(image_data):
    """Process image with TrOCR"""
    try:
        if not trocr_config.is_ready():
            if not trocr_config.initialize_model():
                raise Exception("TrOCR model not initialized")
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        image = preprocess_image(image)
        
        print(f"Processing image of size: {image.size}")
        
        # Process with TrOCR
        with torch.no_grad():
            # Prepare image
            pixel_values = trocr_config.processor(image, return_tensors="pt").pixel_values.to(trocr_config.device)
            
            # Generate text
            generated_ids = trocr_config.model.generate(
                pixel_values,
                max_length=512,  # Adjust based on your needs
                num_beams=5,     # Beam search for better results
                early_stopping=True
            )
            
            # Decode generated text
            generated_text = trocr_config.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"TrOCR extracted text: {generated_text}")
            return generated_text.strip()
    
    except Exception as error:
        print(f"TrOCR processing error: {error}")
        raise Exception(f"OCR processing failed: {str(error)}")

def process_image_in_chunks(image_data, chunk_size=(512, 512), overlap=50):
    """
    Process large images in chunks for better accuracy
    This is useful for documents with multiple lines/sections
    """
    try:
        if not trocr_config.is_ready():
            if not trocr_config.initialize_model():
                raise Exception("TrOCR model not initialized")
        
        image = Image.open(io.BytesIO(image_data))
        image = preprocess_image(image)
        
        width, height = image.size
        chunk_width, chunk_height = chunk_size
        
        # If image is smaller than chunk size, process normally
        if width <= chunk_width and height <= chunk_height:
            return process_with_trocr(image_data)
        
        extracted_texts = []
        
        # Process in chunks
        for y in range(0, height, chunk_height - overlap):
            for x in range(0, width, chunk_width - overlap):
                # Define chunk boundaries
                left = max(0, x)
                top = max(0, y)
                right = min(width, x + chunk_width)
                bottom = min(height, y + chunk_height)
                
                # Skip if chunk is too small
                if right - left < 100 or bottom - top < 100:
                    continue
                
                # Extract chunk
                chunk = image.crop((left, top, right, bottom))
                
                # Convert chunk back to bytes
                chunk_buffer = io.BytesIO()
                chunk.save(chunk_buffer, format='PNG')
                chunk_data = chunk_buffer.getvalue()
                
                # Process chunk
                try:
                    chunk_text = process_with_trocr(chunk_data)
                    if chunk_text.strip():
                        extracted_texts.append(chunk_text)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
        
        # Combine all extracted texts
        combined_text = ' '.join(extracted_texts)
        print(f"Combined text from {len(extracted_texts)} chunks: {combined_text}")
        
        return combined_text
        
    except Exception as e:
        print(f"Chunk processing error: {e}")
        # Fallback to normal processing
        return process_with_trocr(image_data)

def refine_content(content):
    """
    This function takes the OCR content and uses LLM to refine and suggest better responses
    """
    try:
        system_prompt = """
        You are an OCR text correction specialist. Your task is to:
        1. Analyze the given text for spelling mistakes and OCR errors
        2. Provide exactly 3 refined versions of the text
        3. Keep the same meaning and word count as much as possible
        4. Only correct obvious spelling mistakes and OCR misreads
        5. Do not add or remove content, just correct errors
        6. Format must be exactly the same as i asked

        Format your response as JSON:
        {
            "suggestions": [
                "First corrected version",
                "Second corrected version", 
                "Third corrected version"
            ],
            "confidence": "high/medium/low"
        }

        Examples of corrections:
        - "Tever" -> "Fever"
        - "hpadache" -> "headache" 
        - "medcine" -> "medicine"
        - "payn" -> "pain"

        Format must be exactly the same as i asked
        """
        
        human_prompt = f"""
        Please analyze and correct this OCR text: "{content}"
        
        Provide 3 refined versions that correct spelling mistakes while preserving the original meaning.
        """
        
        response = llm1.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        # Try to parse JSON response
        import json
        try:
            # Clean the response content
            content_response = response.content.strip()
            if content_response.startswith('```json'):
                content_response = content_response.replace('```json', '').replace('```', '').strip()
            
            refined_data = json.loads(content_response)
            
            # Ensure we have the right structure
            if 'suggestions' not in refined_data:
                refined_data = {
                    "suggestions": [content.strip(), content.strip(), content.strip()],
                    "confidence": "low"
                }
            
            return refined_data
            
        except Exception as parse_error:
            print(f"JSON parsing failed: {parse_error}")
            print(f"Raw LLM response: {response.content}")
            
            # Extract suggestions manually if JSON parsing fails
            lines = response.content.strip().split('\n')
            suggestions = []
            for line in lines:
                if line.strip() and not line.startswith('{') and not line.startswith('}'):
                    clean_line = line.strip().strip('"').strip(',').strip()
                    if clean_line and not clean_line.startswith('"suggestions"'):
                        suggestions.append(clean_line)
            
            if len(suggestions) < 3:
                suggestions = [content.strip(), content.strip(), content.strip()]
            
            return {
                "suggestions": suggestions[:3],
                "confidence": "medium"
            }
            
    except Exception as e:
        print(f"Error in refine_content: {e}")
        # Return original content as fallback
        return {
            "suggestions": [content.strip(), content.strip(), content.strip()],
            "confidence": "low",
            "error": str(e)
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    trocr_status = "ready" if trocr_config.is_ready() else "not_initialized"
    return jsonify({
        "status": "healthy", 
        "message": "OCR API is running",
        "trocr_status": trocr_status,
        "device": str(trocr_config.device),
        "model": trocr_config.model_name
    })
@app.route('/process-ocr', methods=['POST'])
def process_ocr():
    """
    Process OCR request with TrOCR - supports images and PDFs
    """
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "No data provided"}), 400
        
        image_base64 = request.json['image']
        user_id = request.json.get('user_id', 'anonymous')
        use_chunking = request.json.get('use_chunking', False)
        file_type = request.json.get('file_type', 'image')  # ADD THIS LINE
        
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 to binary data
        try:
            file_data = base64.b64decode(image_base64)
            image_hash = create_image_hash(file_data)
        except Exception as e:
            return jsonify({"error": "Invalid base64 data"}), 400
        
        # ADD THIS SECTION - Process based on file type
        print(f"Processing file type: {file_type}")
        
        if file_type == 'pdf':
            extracted_text = process_pdf_with_trocr(file_data)
        else:
            # Image processing (existing logic)
            if use_chunking:
                extracted_text = process_image_in_chunks(file_data)
            else:
                extracted_text = process_with_trocr(file_data)
        
        print(f"Extracted text: {extracted_text}")
        
        # Apply learned corrections
        learned_corrected_text, applied_corrections = apply_learned_corrections(
            extracted_text, user_id
        )
        print(f"Applied corrections: {applied_corrections}")
        
        # Refine content with LLM
        refined_data = None
        if learned_corrected_text and learned_corrected_text.strip():
            refined_data = refine_content(learned_corrected_text)
            print(f"Refined data: {refined_data}")
        
        response_data = {
            "success": True,
            "original_text": extracted_text or "No text detected",
            "learned_corrected_text": learned_corrected_text,
            "applied_corrections": applied_corrections,
            "refined_data": refined_data,
            "image_hash": image_hash,
            "user_id": user_id,
            "file_type": file_type,  # ADD THIS
            "processing_method": "chunked" if use_chunking else "single",
            "message": "OCR processing completed successfully with TrOCR"
        }
        
        print(f"Sending response: {response_data}")
        return jsonify(response_data)
    
    except Exception as error:
        print(f"OCR processing error: {error}")
        import traceback
        traceback.print_exc()  # ADD THIS for better error tracking
        return jsonify({
            "success": False,
            "error": str(error),
            "message": "TrOCR processing failed"
        }), 500

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """Handle user corrections and learn from them"""
    try:
        data = request.json
        
        if not data or 'original_text' not in data or 'corrected_text' not in data:
            return jsonify({"error": "Missing required fields"}), 400
        
        original_text = data['original_text']
        corrected_text = data['corrected_text']
        user_id = data.get('user_id', 'anonymous')
        image_hash = data.get('image_hash')
        confidence_score = data.get('confidence_score')
        
        # Save the correction
        success = save_user_correction(user_id, original_text, corrected_text, image_hash, confidence_score)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Feedback saved successfully",
                "learned": True
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to save feedback"
            }), 500
        
    except Exception as error:
        print(f"Feedback error: {error}")
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500
# ? related to database and user stats (feedback mechanism)
@app.route('/get-user-stats', methods=['GET'])
def get_user_stats():
    """Get user correction statistics """
    try:
        user_id = request.args.get('user_id', 'anonymous')
        
        if db is None:
            return jsonify({"error": "Database not connected"}), 500
        
        # Count corrections
        correction_count = db.corrections.count_documents({"user_id": user_id})
        
        # Count learned patterns
        pattern_count = db.patterns.count_documents({"user_id": user_id})
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "total_corrections": correction_count,
            "learned_patterns": pattern_count
        })
        
    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

@app.route('/test-trocr', methods=['GET'])
def test_trocr():
    """Test TrOCR model initialization and basic functionality"""
    try:
        if not trocr_config.is_ready():
            init_success = trocr_config.initialize_model()
            if not init_success:
                return jsonify({
                    "success": False,
                    "message": "Failed to initialize TrOCR model"
                }), 500
        
        return jsonify({
            "success": True,
            "message": "TrOCR is working",
            "model_name": trocr_config.model_name,
            "device": str(trocr_config.device),
            "cuda_available": torch.cuda.is_available()
        })
        
    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

@app.route('/test-llm', methods=['GET'])
def test_llm_connection():
    """Test LLM connection"""
    try:
        test_response = refine_content("test message")
        return jsonify({
            "success": True,
            "message": "LLM is working",
            "test_response": test_response
        })
    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

@app.route('/switch-model', methods=['POST'])
def switch_trocr_model():
    """Switch TrOCR model (printed/handwritten/etc.)"""
    try:
        data = request.json
        new_model = data.get('model_name')
        
        if not new_model:
            return jsonify({"error": "No model name provided"}), 400
        
        # Validate model name
        valid_models = [
            'microsoft/trocr-base-printed',
            'microsoft/trocr-base-handwritten',
            'microsoft/trocr-large-printed',
            'microsoft/trocr-large-handwritten'
        ]
        
        if new_model not in valid_models:
            return jsonify({
                "error": f"Invalid model. Valid options: {valid_models}"
            }), 400
        
        # Update configuration
        trocr_config.model_name = new_model
        trocr_config.initialized = False
        trocr_config.processor = None
        trocr_config.model = None
        
        # Initialize new model
        success = trocr_config.initialize_model()
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Switched to model: {new_model}",
                "current_model": trocr_config.model_name
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to initialize new model"
            }), 500
            
    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

# Initialize TrOCR on startup
@app.before_request
def initialize_trocr():
    """Initialize TrOCR model when the app starts"""
    print("Initializing TrOCR model on startup...")
    success = trocr_config.initialize_model()
    if success:
        print("✅ TrOCR initialized successfully on startup")
    else:
        print("❌ Failed to initialize TrOCR on startup")

# For local development
if __name__ == '__main__':
    db_connected = initialize_database()
    
    if not db_connected:
        print("⚠️  Warning: Running without database - corrections won't be saved")
        print("💡 To fix this:")
        print("   1. Install MongoDB: https://docs.mongodb.com/manual/installation/")
        print("   2. Start MongoDB service")
        print("   3. Restart this Flask app")
    
    # Initialize TrOCR
    print("Initializing TrOCR model...")
    trocr_init_success = trocr_config.initialize_model()
    if trocr_init_success:
        print("✅ TrOCR ready for processing")
    else:
        print("❌ TrOCR initialization failed - check dependencies")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

# For Azure deployment (uncomment when deploying)
# if __name__ == '__main__':
#     db_connected = initialize_database()
    
#     if not db_connected:
#         print("⚠️  Warning: Running without database - corrections won't be saved")
    
#     # Initialize TrOCR
#     trocr_init_success = trocr_config.initialize_model()
#     if not trocr_init_success:
#         print("❌ TrOCR initialization failed")
    
#     port = int(os.environ.get("PORT", 5000))
#     app.run(debug=False, host="0.0.0.0", port=port)