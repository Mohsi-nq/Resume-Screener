<p align="center">
  <h1>SmartHire AI: End-to-End Resume Screening Suite</h1>
  <h3>An intelligent NLP pipeline featuring Multi-Class Domain Classification and Semantic Skill-Gap Analytics</h3>
</p>

---

## System Architecture & Core Features

- **Two-Stage Intelligence Pipeline:** Processes raw text streams extracted from multi-page candidate PDF files using customized lexical cleaning array procedures.
- **Supervised Domain Classification:** Utilizes an integrated Scikit-Learn classification model to predict candidate industry specializations with high operational precision.
- **Vector Space Matching:** Computes geometric distance metrics using TF-IDF Vectorization and Cosine Similarity to calculate real-time contextual alignment scores against target job descriptions.
- **AI Skill-Gap Discrepancy Matrix:** Features a built-in automated keyword parsing and token comparison engine that analyzes text arrays to instantly identify missing requirements for talent acquisition teams.
- **Tailored Light UI Canvas:** Deployed an executive, high-contrast user interface engineered entirely with responsive Streamlit layout assets.

##  Technical Stack
- **Language:** Python
- **Machine Learning & NLP:** Scikit-Learn, Feature Extraction Matrices, Joblib
- **Data Engineering:** Pandas, NumPy, PyPDF2, Regular Expressions (Regex)
- **UI Framework:** Streamlit

## Quick Installation & Local Deployment

1. **Clone this repository infrastructure:**
```bash
   git clone [https://github.com/YOUR_USERNAME/Resume-Screener.git](https://github.com/YOUR_USERNAME/Resume-Screener.git)
   cd Resume-Screener
Establish and activate an isolated virtual environment:

Bash
   python -m venv venv
   # Windows Activation:
   venv\Scripts\activate
   # macOS/Linux Activation:
   source venv/bin/activate
Install exact project dependencies:

Bash
   pip install -r requirements.txt
Initialize the local Streamlit dashboard server:

Bash
   streamlit run app.py
