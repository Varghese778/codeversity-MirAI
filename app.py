"""
MirAI Flask Backend - Connects the website to the ML models
Run: python app.py
Open: http://localhost:5000
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

app = Flask(__name__, static_folder='.')
CORS(app)

# Try to load the real model
try:
    from mirai_inference import MirAI_System
    mirai = MirAI_System(artifacts_dir='backend/models')
    USE_REAL_MODEL = True
    print("✅ MirAI ML models loaded successfully!")
except Exception as e:
    USE_REAL_MODEL = False
    print(f"⚠️ Could not load models: {e}")
    print("   Using mock predictions. Run model.ipynb to generate models.")


def mock_prediction(data):
    """Fallback mock prediction when models aren't available"""
    score = 15
    age = int(data.get('AGE', 65))
    if age > 75: score += 20
    elif age > 65: score += 10
    faq = float(data.get('FAQ', 0))
    score += faq * 1.5
    ecog = float(data.get('EcogPtMem', 1))
    score += (ecog - 1) * 10
    apoe4 = int(data.get('APOE4', 0))
    score += apoe4 * 12
    ptau = float(data.get('PTAU', 0))
    if ptau > 0.6: score += 20
    elif ptau > 0.3: score += 10
    score = min(round(score), 100)
    
    return {
        'final_risk_score': score / 100,
        'stage1_prob': min(score * 0.8, 95) / 100,
        'stage2_prob': min(score * 0.9, 98) / 100,
        'stage3_prob': score / 100,
        'stage1_risk': 'Low' if score < 30 else ('Elevated' if score < 60 else 'High'),
        'stage2_risk': 'Elevated' if apoe4 > 0 else 'Low',
        'stage3_risk': 'Elevated' if ptau > 0.5 else ('Normal' if ptau > 0 else 'Not Tested'),
        'top_factors': [
            f"FAQ Score: {faq}",
            f"APOE4 Count: {apoe4}",
            f"pTau-217: {ptau if ptau > 0 else 'Not provided'}"
        ]
    }


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        patient_data = {
            'AGE': float(data.get('age', 65)),
            'PTGENDER': data.get('gender', 'Male'),
            'PTEDUCAT': float(data.get('education', 16)),
            'FAQ': float(data.get('faq', 0)),
            'EcogPtMem': float(data.get('ecogMem', 1)),
            'EcogPtTotal': float(data.get('ecogTotal', 1)),
            'APOE4': parse_apoe4(data.get('genotype', '')),
            'PTAU': float(data.get('ptau217', 0) or 0),
            'ABETA42': float(data.get('ab42', 0) or 0),
            'ABETA40': float(data.get('ab40', 0) or 0),
            'NFL': float(data.get('nfl', 0) or 0)
        }
        
        if USE_REAL_MODEL:
            result = mirai.predict(patient_data)
        else:
            result = mock_prediction(patient_data)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'model_type': 'real' if USE_REAL_MODEL else 'mock'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def parse_apoe4(genotype):
    if not genotype:
        return 0
    return genotype.count('4')


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🧠 MirAI Alzheimer's Screening Server")
    print("="*50)
    print(f"Model: {'✅ Real ML Models' if USE_REAL_MODEL else '⚠️ Mock Mode'}")
    print("URL: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
