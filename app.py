import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import clean_resume, extract_text_from_pdf

# 1. PAGE SETUP
st.set_page_config(
    page_title="AI Resume Screener",  
    layout="wide"
)

# 2. LOAD TRAINED MODELS
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        model = joblib.load("models/resume_classifier.pkl")
        return vectorizer, model
    except FileNotFoundError:
        st.error("Model files missing! Please run 'python src/train.py' in your terminal first.")
        return None, None

vectorizer, model = load_models()

# 3. APP HEADER
st.markdown("<h1 style='text-align: center;'>SmartHire AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>Instantly sort resumes by job category and rank them by matching score</p>", unsafe_allow_html=True)
st.markdown("---")

if vectorizer and model:
    # Split the screen into two clean side-by-side columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Job Description")
        jd_input = st.text_area(
            "Paste your target job requirements here:", 
            height=230,
            placeholder="Example: Looking for a Data Scientist with Python, SQL, and Machine Learning experience..."
        )
        
    with col2:
        st.header("Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload applicant resume PDFs here (Supports multiple files):", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
    st.markdown("---")
    
    # 4. RUN CALCULATIONS WHEN BUTTON IS CLICKED
    if st.button("Analyze & Rank Resumes", use_container_width=True):
        if not jd_input or not uploaded_files:
            st.warning("Please make sure you have pasted a job description and uploaded at least one resume PDF.")
        else:
            payload = []
            
            with st.spinner("Analyzing resumes... Please wait..."):
                for file in uploaded_files:
                    # Extract and clean text from the PDF
                    raw_text = extract_text_from_pdf(file)
                    cleaned_text = clean_resume(raw_text)
                    
                    # Step A: Predict the job category (Data Science, HR, Web Design, etc.)
                    vec_input = vectorizer.transform([cleaned_text])
                    domain_prediction = model.predict(vec_input)[0]
                    
                    # Step B: Calculate matching score against the Job Description
                    cleaned_jd = clean_resume(jd_input)
                    local_vectorizer = TfidfVectorizer(stop_words="english")
                    matrix = local_vectorizer.fit_transform([cleaned_text, cleaned_jd])
                    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
                    
                    payload.append({
                        "Candidate File Name": file.name,
                        "Predicted Job Category": domain_prediction,
                        "Matching Score (%)": round(similarity * 100, 2)
                    })
                
            # Convert results to a table and sort by highest matching score
            results_df = pd.DataFrame(payload).sort_values(
                by="Matching Score (%)", 
                ascending=False
            ).reset_index(drop=True)
            
            # 5. DISPLAY RESULTS
            st.header("Screening Results Leaderboard")
            
            # Show summary numbers on top
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Total Resumes Processed", len(payload))
            metric_col2.metric("Highest Match Score", f"{results_df.iloc[0]['Matching Score (%)']}%")
            
            st.write("") # Spacer
            
            # Display the data table
            st.dataframe(results_df, use_container_width=True)
            st.success("Analysis complete! Candidates ranked successfully from best match to lowest match.")