from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import tempfile

app = Flask(__name__)
CORS(app) # Elementor/WordPress se request allow karne ke liye

# AI Model Load kar rahe hain (Base model fast hota hai)
print("Loading AI Model...")
model = whisper.load_model("base")
print("Model Loaded!")

def format_timestamp(seconds):
    """Seconds ko SRT format (HH:MM:SS,mmm) mein convert karta hai"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

@app.route('/api/generate', methods=['POST'])
def generate_captions():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file uploaded"}), 400

    file = request.files['video']
    
    # Video ko temporary file mein save karo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        file.save(temp_video.name)
        temp_video_path = temp_video.name

    try:
        # Whisper AI se Audio ko Text mein convert karna
        # Note: Whisper Hindi audio ko English alphabets (Hinglish-ish) mein transcribe kar sakta hai
        result = model.transcribe(temp_video_path)
        
        captions = []
        for segment in result["segments"]:
            captions.append({
                "start": format_timestamp(segment["start"]),
                "end": format_timestamp(segment["end"]),
                "text": segment["text"].strip()
            })

        # Temporary video delete kar do memory bachane ke liye
        os.remove(temp_video_path)
        
        return jsonify({
            "success": True, 
            "captions": captions
        })

    except Exception as e:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Server start on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)