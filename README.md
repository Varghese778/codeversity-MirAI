# 🧠 MirAI - Early Alzheimer's Screening System

**AI-powered 3-tier screening that catches risk before symptoms appear.**

![MirAI](https://img.shields.io/badge/MirAI-Alzheimer's%20Screening-5E4BFF)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌐 Live Demo

**Static Demo:** [https://varghese778.github.io/codeversity-MirAI/](https://varghese778.github.io/codeversity-MirAI/)

> ⚠️ The live demo uses client-side mock calculations. For real ML model predictions, follow the local setup instructions below.

---

## 🚀 Quick Start (Local with Real ML Models)

### Prerequisites
- Python 3.8+
- pip

### Step 1: Clone the Repository
```bash
git clone https://github.com/Varghese778/codeversity-MirAI.git
cd codeversity-MirAI
```

### Step 2: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 3: Train the ML Models
```bash
jupyter notebook model.ipynb
```
Run all cells in the notebook. This will:
- Load ADNI data from `data/` folder
- Train 3-stage cascade models (Clinical → Genetic → Biomarker)
- Save model artifacts to `models/` folder

### Step 4: Run the Flask Server
```bash
cd ..
python app.py
```

### Step 5: Open in Browser
Navigate to: **http://localhost:5000**

---

## 📁 Project Structure

```
codeversity-MirAI/
├── index.html              # Homepage
├── assessment.html         # Risk assessment wizard
├── results.html            # Results visualization
├── app.py                  # Flask backend server
├── assets/                 # CSS, JS, images, vendor libs
│   ├── css/main.css        # MirAI color scheme
│   └── vendor/             # Bootstrap, AOS, etc.
├── backend/
│   ├── model.ipynb         # Jupyter notebook for training
│   ├── mirai_inference.py  # Inference script
│   ├── requirements.txt    # Python dependencies
│   ├── data/               # ADNI CSV files
│   │   ├── ADNIMERGE_01Feb2026.csv
│   │   ├── APOERES_01Feb2026.csv
│   │   ├── UPENN_PLASMA_*.csv
│   │   └── ...
│   └── models/             # Trained model artifacts (generated)
│       ├── stage1_pipeline.pkl
│       ├── stage2_pipeline.pkl
│       ├── stage3_pipeline.pkl
│       └── *.json
└── README.md
```

---

## 🔬 The 3-Stage Pipeline

| Stage | Name | Features | Purpose |
|-------|------|----------|---------|
| 1 | **Clinical Screening** | Age, Gender, Education, FAQ, ECog | Non-invasive initial triage |
| 2 | **Genetic Stratification** | Stage1_Prob + APOE ε4 | Risk refinement using genetics |
| 3 | **Biomarker Confirmation** | Stage2_Prob + pTau-217, Aβ42, NfL | Blood-based pathology detection |

### Model Performance (on ADNI data)
- Stage 1: AUC ~0.87
- Stage 2: AUC ~0.88
- Stage 3: AUC ~0.93

---

## 🛠️ API Reference

### POST `/api/predict`

**Request Body:**
```json
{
  "age": 72,
  "gender": "Female",
  "education": 16,
  "faq": 8,
  "ecogMem": 2.5,
  "ecogTotal": 2.0,
  "genotype": "3/4",
  "ptau217": 0.45,
  "ab42": 15.2,
  "ab40": 180.5,
  "nfl": 22.0
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "final_risk_score": 0.62,
    "final_risk_category": "Elevated",
    "stage1_risk": "Elevated",
    "stage2_risk": "Elevated",
    "stage3_risk": "Elevated",
    "top_factors": [
      "FAQ Score: 8",
      "APOE4 Count: 1",
      "pTau-217: 0.45"
    ]
  },
  "model_type": "real"
}
```

---

## 👥 Team

**Team Break&Build**
- 📞 +91 9531975283
- 🔗 GitHub: [@Varghese778](https://github.com/Varghese778)

---

## 📚 Research Foundation

MirAI's methodology is grounded in established Alzheimer's research:

- **Marshal Folstein** - MMSE Developer
- **Allen D. Roses** - APOE ε4 Pioneer
- **Oskar Hansson** - pTau-217 Researcher

Data source: [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/)

---

## ⚠️ Disclaimer

This is a **screening tool**, not a diagnostic. Results indicate relative risk based on inputs. Always consult a qualified healthcare provider for clinical evaluation.

---

## 📄 License

MIT License - Feel free to use and modify for research and educational purposes.
