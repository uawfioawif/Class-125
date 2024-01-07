from flask import Flask, jsonify, request
from model import getPrediction

app = Flask(__name__)

@app.route('/predict-digit', methods=['POST'])
def predictDigit():
  image = request.files.get('digit')
  prediction = getPrediction(image)
  return jsonify({
    'prediction': prediction
  }), 200

if __name__ == '__main__':
  app.run(debug=True)
