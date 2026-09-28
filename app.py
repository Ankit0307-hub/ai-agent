import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request, jsonify
from agent import run_agent, run_voice_agent

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No request data received."
            }), 400

        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "error": "Please enter a question."
            }), 400

        result = run_agent(question)

        if result.get("error"):
            return jsonify({
                "error": result["error"]
            }), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/voice", methods=["POST"])
def voice():
    try:
        if "audio" not in request.files:
            return jsonify({
                "error": "No audio recording received."
            }), 400

        audio_file = request.files["audio"]

        if audio_file.filename == "":
            return jsonify({
                "error": "Empty audio recording."
            }), 400

        audio_bytes = audio_file.read()

        if not audio_bytes:
            return jsonify({
                "error": "Audio recording is empty."
            }), 400

        result = run_voice_agent(
            audio_bytes,
            audio_file.mimetype or "audio/webm"
        )

        if result.get("error"):
            return jsonify({
                "error": result["error"]
            }), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )