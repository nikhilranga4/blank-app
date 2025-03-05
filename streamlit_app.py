import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text.strip()

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities * 100  # Convert to percentage

# Streamlit UI enhancements
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

# Header section
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>📄 AI Resume Screening & Candidate Ranking System</h1>
    <p style='text-align: center; font-size: 18px;'>Upload resumes and enter a job description to rank candidates based on relevance.</p>
    <hr>
    """, unsafe_allow_html=True)

# Job description input
st.subheader("Job Description")
job_description = st.text_area("Enter the job description here:", height=150)

# File uploader
st.subheader("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.subheader("Candidate Ranking")
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)

    # Display results in a styled DataFrame
    results = pd.DataFrame({"Resume Name": [file.name for file in uploaded_files], "Score (%)": scores})
    results = results.sort_values(by="Score (%)", ascending=False)
    
    # Add a color gradient to highlight scores
    st.dataframe(results.style.format({"Score (%)": "{:.2f}%"}).background_gradient(cmap="Blues"))

    st.success("Ranking completed! Candidates are listed based on relevance to the job description.")
else:
    st.info("Please enter a job description and upload resumes to proceed.")
