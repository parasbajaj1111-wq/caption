from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import tempfile

app = Flask(__name__)
CORS(app)

print("Loading AI Model (Tiny)...")
# 'base' ki jagah 'tiny' use kar rahe hain taaki 512MB RAM mein fit aa jaye
model = whisper.load_model("tiny")
print("Model Loaded!")

def format_timestamp(seconds):
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
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        file.save(temp_video.name)
        temp_video_path = temp_video.name

    try:
        # fp16=False karna zaroori hai bina GPU wale (free) server par
        result = model.transcribe(temp_video_path, fp16=False)
        
        captions = []
        for segment in result["segments"]:
            captions.append({
                "start": format_timestamp(segment["start"]),
                "end": format_timestamp(segment["end"]),
                "text": segment["text"].strip()
            })

        os.remove(temp_video_path)
        return jsonify({"success": True, "captions": captions})

    except Exception as e:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Render apna khud ka port assign karta hai, isliye port dynamic kiya hai
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)