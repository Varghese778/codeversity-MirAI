"""
MirAI Inference Script - 3-Stage Cascade Model for Alzheimer's Screening
"""
import os
import json
import pickle
import numpy as np
import pandas as pd


class MirAI_System:
    """
    MirAI 3-Stage Cascade Prediction System
    
    Stages:
        1. Clinical Screening (Age, Gender, Education, FAQ, ECog)
        2. Genetic Stratification (Stage1 + APOE4)
        3. Biomarker Confirmation (Stage2 + pTau217, Aβ42, Aβ40, NfL)
    """
    
    def __init__(self, artifacts_dir="models"):
        """Initialize by loading all trained artifacts."""
        self.artifacts_dir = artifacts_dir
        self.artifacts = {}
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load all model artifacts from disk."""
        artifact_files = [
            'stage1_pipeline.pkl',
            'stage2_pipeline.pkl', 
            'stage3_pipeline.pkl',
            'stage1_features.json',
            'stage2_features.json',
            'stage3_features.json',
            'stage1_threshold.json',
            'stage2_threshold.json'
        ]
        
        for fname in artifact_files:
            path = os.path.join(self.artifacts_dir, fname)
            if os.path.exists(path):
                if fname.endswith('.pkl'):
                    with open(path, 'rb') as f:
                        self.artifacts[fname.replace('.pkl', '')] = pickle.load(f)
                elif fname.endswith('.json'):
                    with open(path, 'r') as f:
                        self.artifacts[fname.replace('.json', '')] = json.load(f)
        
        if not self.artifacts:
            raise FileNotFoundError(f"No artifacts found in {self.artifacts_dir}. Run model.ipynb first.")
    
    def preprocess_gender(self, gender):
        """Convert gender to numeric (Male=1, Female=0)."""
        if isinstance(gender, str):
            return 1 if gender.lower() == 'male' else 0
        return gender
    
    def preprocess_apoe4(self, genotype):
        """Convert APOE genotype to count of ε4 alleles."""
        if isinstance(genotype, str):
            return genotype.count('4')
        return genotype if genotype else 0
    
    def predict(self, patient_data):
        """
        Run 3-stage cascade prediction.
        
        Args:
            patient_data: dict with keys like AGE, PTGENDER, FAQ, etc.
            
        Returns:
            dict with predictions and risk levels
        """
        # Preprocess
        patient_data['PTGENDER'] = self.preprocess_gender(patient_data.get('PTGENDER', 'Male'))
        patient_data['APOE4'] = self.preprocess_apoe4(patient_data.get('APOE4', 0))
        
        # Stage 1: Clinical
        stage1_features = self.artifacts.get('stage1_features', 
            ['AGE', 'PTGENDER', 'PTEDUCAT', 'FAQ', 'EcogPtMem', 'EcogPtTotal'])
        X1 = pd.DataFrame([{f: patient_data.get(f, 0) for f in stage1_features}])
        stage1_prob = self.artifacts['stage1_pipeline'].predict_proba(X1)[0, 1]
        stage1_threshold = self.artifacts.get('stage1_threshold', {}).get('threshold', 0.5)
        stage1_risk = 'Elevated' if stage1_prob >= stage1_threshold else 'Low'
        
        # Stage 2: Genetic
        stage2_features = self.artifacts.get('stage2_features', ['Stage1_Prob', 'APOE4'])
        X2_data = {'Stage1_Prob': stage1_prob, 'APOE4': patient_data.get('APOE4', 0)}
        X2 = pd.DataFrame([X2_data])
        stage2_prob = self.artifacts['stage2_pipeline'].predict_proba(X2)[0, 1]
        stage2_threshold = self.artifacts.get('stage2_threshold', {}).get('threshold', 0.5)
        stage2_risk = 'Elevated' if stage2_prob >= stage2_threshold else 'Low'
        
        # Stage 3: Biomarker
        stage3_features = self.artifacts.get('stage3_features', 
            ['Stage2_Prob', 'PTAU', 'ABETA42', 'ABETA40', 'NFL'])
        X3_data = {
            'Stage2_Prob': stage2_prob,
            'PTAU': patient_data.get('PTAU', 0),
            'ABETA42': patient_data.get('ABETA42', 0),
            'ABETA40': patient_data.get('ABETA40', 0),
            'NFL': patient_data.get('NFL', 0)
        }
        X3 = pd.DataFrame([X3_data])
        stage3_prob = self.artifacts['stage3_pipeline'].predict_proba(X3)[0, 1]
        
        # Final risk categorization
        if stage3_prob >= 0.7:
            final_risk = 'High'
        elif stage3_prob >= 0.4:
            final_risk = 'Elevated'
        else:
            final_risk = 'Low'
        
        return {
            'final_risk_score': float(stage3_prob),
            'final_risk_category': final_risk,
            'stage1_prob': float(stage1_prob),
            'stage2_prob': float(stage2_prob),
            'stage3_prob': float(stage3_prob),
            'stage1_risk': stage1_risk,
            'stage2_risk': stage2_risk,
            'stage3_risk': final_risk,
            'top_factors': [
                f"FAQ Score: {patient_data.get('FAQ', 'N/A')}",
                f"APOE4 Count: {patient_data.get('APOE4', 0)}",
                f"pTau-217: {patient_data.get('PTAU', 'Not provided')}"
            ]
        }


if __name__ == '__main__':
    # Test with sample data
    print("MirAI Inference Test")
    print("="*50)
    
    sample_patient = {
        'AGE': 72,
        'PTGENDER': 'Female',
        'PTEDUCAT': 16,
        'FAQ': 8,
        'EcogPtMem': 2.5,
        'EcogPtTotal': 2.0,
        'APOE4': 1,
        'PTAU': 0.45,
        'ABETA42': 15.2,
        'ABETA40': 180.5,
        'NFL': 22.0
    }
    
    try:
        mirai = MirAI_System(artifacts_dir='models')
        result = mirai.predict(sample_patient)
        print(f"Risk Score: {result['final_risk_score']*100:.1f}%")
        print(f"Risk Category: {result['final_risk_category']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo generate model artifacts, run the Jupyter notebook first:")
        print("  jupyter notebook model.ipynb")
