from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import time
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
from configration import  llm1
from Database import save_user_correction, get_learned_patterns, apply_learned_corrections,create_image_hash, initialize_database, db

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Azure OCR configuration
AZURE_ENDPOINT = os.getenv('REACT_APP_AZURE_ENDPOINT')
AZURE_KEY = os.getenv('REACT_APP_AZURE_KEY')

def process_with_azure_ocr(image_data):
    """Process image with Azure OCR API"""
    try:
        # Step 1: Send image for analysis
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_KEY,
            "Content-Type": "application/octet-stream",
        }
        
        response = requests.post(
            f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze",
            headers=headers,
            data=image_data
        )
        
        if not response.ok:
            raise Exception(f"Azure OCR request failed: {response.status_code}")
        
        # Get operation-location to poll results
        operation_location = response.headers.get("operation-location")
        print(f"Operation location: {operation_location}")
        
        if not operation_location:
            raise Exception("No operation location returned from Azure")
        
        # Step 2: Poll until result is ready
        poll_headers = {"Ocp-Apim-Subscription-Key": AZURE_KEY}
        
        while True:
            poll_response = requests.get(operation_location, headers=poll_headers)
            result = poll_response.json()
            
            if result.get("status") == "succeeded":
                # Extract text from results
                extracted_text = ""
                if "analyzeResult" in result and "readResults" in result["analyzeResult"]:
                    for page in result["analyzeResult"]["readResults"]:
                        for line in page.get("lines", []):
                            extracted_text += line.get("text", "") + " "
                
                return extracted_text.strip()
            
            elif result.get("status") == "failed":
                raise Exception("OCR processing failed")
            
            # Wait before polling again
            time.sleep(1)
    
    except Exception as error:
        print(f"Azure OCR error: {error}")
        raise Exception(f"OCR processing failed: {str(error)}")

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
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            refined_data = json.loads(content)
            
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
    return jsonify({"status": "healthy", "message": "OCR API is running"})

@app.route('/process-ocr', methods=['POST'])
def process_ocr():
    """
    Process OCR request with learning integration
    """
    try:
        # Check if request has JSON data
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        
        # Get base64 image data and user ID
        image_base64 = request.json['image']
        user_id = request.json.get('user_id', 'anonymous')
        
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 to binary data
        try:
            image_data = base64.b64decode(image_base64)
            image_hash = create_image_hash(image_data)
        except Exception as e:
            return jsonify({"error": "Invalid base64 image data"}), 400
        
        # Process with Azure OCR
        extracted_text = process_with_azure_ocr(image_data)
        print(f"Extracted text: {extracted_text}")
        
        # Apply learned corrections BEFORE LLM processing
        learned_corrected_text, applied_corrections = apply_learned_corrections(extracted_text, user_id)
        print(f"Applied corrections: {applied_corrections}")
        
        # Refine content with LLM using the learned-corrected text
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
            "message": "OCR processing completed successfully"
        }
        
        print(f"Sending response: {response_data}")
        return jsonify(response_data)
    
    except Exception as error:
        print(f"OCR processing error: {error}")
        return jsonify({
            "success": False,
            "error": str(error),
            "message": "OCR processing failed"
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
            # Update patterns (we'll implement this later)
            # update_patterns(original_text, corrected_text, user_id)
            
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

@app.route('/get-user-stats', methods=['GET'])
def get_user_stats():
    """Get user correction statistics"""
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

@app.route('/test-azure', methods=['GET'])
def test_azure_connection():
    """Test Azure API connection"""
    try:
        if not AZURE_ENDPOINT or not AZURE_KEY:
            return jsonify({
                "success": False,
                "message": "Azure credentials not configured"
            }), 500
        
        return jsonify({
            "success": True,
            "message": "Azure credentials are configured",
            "endpoint_configured": bool(AZURE_ENDPOINT),
            "key_configured": bool(AZURE_KEY)
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


# using this in local and working fine
# if __name__ == '__main__':

#     db_connected = initialize_database()
    
#     if not db_connected:
#         print("⚠️  Warning: Running without database - corrections won't be saved")
#         print("💡 To fix this:")
#         print("   1. Install MongoDB: https://docs.mongodb.com/manual/installation/")
#         print("   2. Start MongoDB service")
#         print("   3. Restart this Flask app")

#     # Check for required environment variables
#     if not AZURE_ENDPOINT or not AZURE_KEY:
#         print("Warning: Azure credentials not found in environment variables")
#         print("Please set AZURE_ENDPOINT and AZURE_KEY in your .env file")
    
#     app.run(debug=True, host='0.0.0.0', port=5000)


# using this becouse not working on azure
if __name__ == '__main__':

    db_connected = initialize_database()
    
    if not db_connected:
        print("⚠️  Warning: Running without database - corrections won't be saved")
        print("💡 To fix this:")
        print("   1. Install MongoDB: https://docs.mongodb.com/manual/installation/")
        print("   2. Start MongoDB service")
        print("   3. Restart this Flask app")

    # Check for required environment variables
    if not AZURE_ENDPOINT or not AZURE_KEY:
        print("Warning: Azure credentials not found in environment variables")
        print("Please set AZURE_ENDPOINT and AZURE_KEY in your .env file")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
