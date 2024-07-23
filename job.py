import streamlit as st
import pandas as pd
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data_path = "./Book2.csv"
df = pd.read_csv(data_path)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# Page 1: Upload Resume and Find Job Details
def page1():
    st.title("Resume Job Matcher")
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            try:
                resume_text = uploaded_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Error reading text file: {e}")
                return
        
        # Check if the resume text was successfully extracted
        if not resume_text:
            st.error("No text could be extracted from the uploaded resume.")
            return
        
        # Combine the resume text with the job descriptions for TF-IDF vectorization
        job_descriptions = df['Job Description'].tolist()
        corpus = [resume_text] + job_descriptions
        
        # Debugging: print the corpus to the Streamlit app
        st.write("Corpus contents:", corpus[:5])  # Display first 5 items for debugging
        
        # Vectorize the text using TF-IDF
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError as e:
            st.error(f"Error in fit_transform: {e}")
            return
        
        # Compute cosine similarity between the resume and all job descriptions
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Get the top matching job descriptions
        top_matches = cosine_similarities.argsort()[-5:][::-1]
        
        if cosine_similarities[top_matches[0]] > 0:  # Check if there are any matches
            for idx in top_matches:
                job = df.iloc[idx]
                st.write(f"**Job Title:** {job['Job Title']}")
                st.write(f"**Company:** {job['Company']}")
                st.write(f"**Qualifications:** {job['Qualifications']}")
                st.write(f"**Preference:** {job['Preference']}")
                st.write(f"**Job Description:** {job['Job Description']}")
                st.write("---")
        else:
            st.write("No matching jobs found.")

# Page 2: Search Job by Title
def page2():
    st.title("Job Details Search")
    job_title = st.text_input("Enter Job Title")
    
    if job_title:
        matching_job = df[df['Job Title'].str.contains(job_title, case=False, na=False)]
        
        if not matching_job.empty:
            job = matching_job.iloc[0]
            st.write(f"**Experience:** {job['Experience']}")
            st.write(f"**Qualifications:** {job['Qualifications']}")
            st.write(f"**Salary Range:** {job['Salary Range']}")
            st.write(f"**Work Type:** {job['Work Type']}")
            st.write(f"**Job Posting Date:** {job['Job Posting Date']}")
            st.write(f"**Preference:** {job['Preference']}")
            st.write(f"**Job Title:** {job['Job Title']}")
            st.write(f"**Role:** {job['Role']}")
            st.write(f"**Job Description:** {job['Job Description']}")
            st.write(f"**Skills:** {job['skills']}")   
            st.write(f"**Responsibilities:** {job['Responsibilities']}")
            st.write(f"**Company:** {job['Company']}")
        else:
            st.write("No matching job found.")

# Streamlit app with multiple pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Resume Job Matcher", "Job Details Search"])

if page == "Resume Job Matcher":
    page1()
else:
    page2()
