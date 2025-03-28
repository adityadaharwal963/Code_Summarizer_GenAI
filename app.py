# Import necessary libraries
from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv
from generateSummary import summary

load_dotenv()

app = Flask(__name__)

# Configure your Gemini API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')


@app.route('/api/generate', methods=['POST'])
def generate_text():
    """
    Endpoint to generate text using the Gemini API.
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        response = model.generate_content(prompt)
        return jsonify({'text': response.text.strip()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stream', methods=['POST'])
def generate_stream():
    """
    Endpoint to generate a stream of text using the Gemini API.
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        response_stream = model.generate_content(prompt, stream=True)
        response_text = []

        for chunk in response_stream:
            if chunk.text:
                response_text.append(chunk.text.strip())

        return jsonify({'text': " ".join(response_text)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """
    Endpoint to retrieve the available models.
    """
    try:
        available_models = [model.name for model in genai.list_models()]
        return jsonify({'models': available_models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/summary', methods=['POST'])
def get_summary():
    """
    Endpoint to generate a structured summary of code using the Gemini API.
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        response_data = summary(prompt)

        # Ensure valid JSON is returned
        if isinstance(response_data, dict):
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Invalid response format from summary function'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Ensure your desired port.
