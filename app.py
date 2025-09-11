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
from configration import llm, llm1

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
    Process OCR request with LLM refinement
    Expects: JSON with base64 encoded image data
    Returns: JSON with original text and refined suggestions
    """
    try:
        # Check if request has JSON data
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        
        # Get base64 image data
        image_base64 = request.json['image']
        
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 to binary data
        try:
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            return jsonify({"error": "Invalid base64 image data"}), 400
        
        # Process with Azure OCR
        extracted_text = process_with_azure_ocr(image_data)
        print(f"Extracted text: {extracted_text}")
        
        # Refine content with LLM if text is found
        refined_data = None
        if extracted_text and extracted_text.strip():
            refined_data = refine_content(extracted_text)
            print(f"Refined data: {refined_data}")
        
        response_data = {
            "success": True,
            "original_text": extracted_text or "No text detected",
            "text": extracted_text,  # Keep for backward compatibility
            "refined_data": refined_data,
            "message": "OCR processing completed successfully"
        }
        
        print(f"Sending response: {response_data}")  # Debug print
        return jsonify(response_data)
    
    except Exception as error:
        print(f"OCR processing error: {error}")
        return jsonify({
            "success": False,
            "error": str(error),
            "message": "OCR processing failed"
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

if __name__ == '__main__':
    # Check for required environment variables
    if not AZURE_ENDPOINT or not AZURE_KEY:
        print("Warning: Azure credentials not found in environment variables")
        print("Please set AZURE_ENDPOINT and AZURE_KEY in your .env file")
    
    app.run(debug=True, host='0.0.0.0', port=5000)