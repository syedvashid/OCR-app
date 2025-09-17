                  
# Add these imports at the top
from pymongo import MongoClient
from datetime import datetime
import hashlib
from bson import ObjectId
import os 

# Global database variable
db = None

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI")
# MONGO_URI = os.getenv("MONGO_URI_CLOUD")

DATABASE_NAME = os.getenv("DATABASE_NAME")


def initialize_database():
    """Initialize MongoDB connection"""
    global db
    try:
        mongo_client = MongoClient(MONGO_URI)
        # Test connection
        mongo_client.admin.command('ping')
        db = mongo_client[DATABASE_NAME]
        print("✅ MongoDB connected successfully")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("💡 Make sure MongoDB is running on your system")
        db = None
        return False

#COre funtion for storing the data in datbase 
def save_user_correction(user_id, original_text, corrected_text, image_hash=None, confidence_score=None):
    """Save user correction to MongoDB"""
    try:
        if db is None:
            print("❌ Database not connected, cannot save correction")
            return False
            
        correction_data = {
            "user_id": user_id,
            "original_text": original_text,
            "corrected_text": corrected_text,
            "image_hash": image_hash,
            "confidence_score": confidence_score,
            "timestamp": datetime.utcnow(),
            "processed": False
        }
        
        result = db.corrections.insert_one(correction_data)
        print(f"✅ Correction saved with ID: {result.inserted_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving correction: {e}")
        return False

def get_learned_patterns(user_id=None):
    """Get learned patterns from MongoDB"""
    try:
        if db is None:
            print("❌ Database not connected, returning empty patterns")
            return []
            
        query = {}
        if user_id:
            query["user_id"] = user_id
            
        patterns = list(db.patterns.find(query).sort("frequency", -1))
        return patterns
        
    except Exception as e:
        print(f"❌ Error fetching patterns: {e}")
        return []

def apply_learned_corrections(text, user_id=None):
    """Apply learned corrections to text"""
    try:
        if db is None:
            print("💡 Database not connected, skipping learned corrections")
            return text, []
            
        patterns = get_learned_patterns(user_id)
        corrected_text = text
        applied_corrections = []
        
        for pattern in patterns:
            if pattern.get('frequency', 0) >= 2:  # Only apply if seen multiple times
                error_pattern = pattern.get('error_pattern', '')
                correct_pattern = pattern.get('correct_pattern', '')
                
                if error_pattern in corrected_text:
                    corrected_text = corrected_text.replace(error_pattern, correct_pattern)
                    applied_corrections.append({
                        'from': error_pattern,
                        'to': correct_pattern,
                        'frequency': pattern.get('frequency', 0),
                        'confidence': min(pattern.get('frequency', 0) * 0.1, 0.9)
                    })
        
        return corrected_text, applied_corrections
        
    except Exception as e:
        print(f"❌ Error applying corrections: {e}")
        return text, []
    
def create_image_hash(image_data):
    """Create hash for image data"""
    return hashlib.md5(image_data).hexdigest()[:16] 
