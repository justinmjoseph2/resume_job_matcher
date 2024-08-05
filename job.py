import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the dataset
data_path = "./Book1.csv"
df = pd.read_csv(data_path)

# Function to process text and remove stopwords
def process_text(text):
    words = text.lower().split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return words

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Page 1: Upload Resume and Find Job Details
def page1():
    st.title("Resume Job Matcher")
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")
        
        resume_words = set(process_text(resume_text))
        
        matching_jobs = []
        
        for idx, row in df.iterrows():
            skills_education_text = f"{str(row.get('Skills', ''))} {str(row.get('Education', ''))}"
            skills_education_words = set(process_text(skills_education_text))
            if resume_words & skills_education_words:
                matching_jobs.append(row)
        
        if matching_jobs:
            for job in matching_jobs:
                st.write(f"**Job Title:** {job['Job Title']}")
                st.write(f"**Company:** {job['Company']}")
                st.write(f"**Qualifications:** {job['Qualifications']}")
                st.write(f"**Preference:** {job['Preference']}")
                st.write(f"**Job Description:** {job['Job Description']}")
                st.write("---")
        else:
            st.write("No matching jobs found.")

# Page 2: Search Job by Title and Edit
def page2():
    st.title("Job Details Search and Edit")
    job_title = st.text_input("Enter Job Title")
    
    if job_title:
        matching_job = df[df['Job Title'].str.contains(job_title, case=False, na=False)]
        
        if not matching_job.empty:
            job = matching_job.iloc[0]
            
            # Editable text boxes for each field
            experience = st.text_area("Experience", job['Experience'])
            qualifications = st.text_area("Qualifications", job['Qualifications'])
            salary_range = st.text_area("Salary Range", job['Salary Range'])
            work_type = st.text_area("Work Type", job['Work Type'])
            job_posting_date = st.text_area("Job Posting Date", job['Job Posting Date'])
            preference = st.text_area("Preference", job['Preference'])
            job_title = st.text_area("Job Title", job['Job Title'])
            role = st.text_area("Role", job['Role'])
            job_description = st.text_area("Job Description", job['Job Description'])
            skills = st.text_area("Skills", job['Skills'])
            responsibilities = st.text_area("Responsibilities", job['Responsibilities'])
            company = st.text_area("Company", job['Company'])
            
            if st.button("Save Changes"):
                new_record = {
                    'Experience': experience,
                    'Qualifications': qualifications,
                    'Salary Range': salary_range,
                    'Work Type': work_type,
                    'Job Posting Date': job_posting_date,
                    'Preference': preference,
                    'Job Title': job_title,
                    'Role': role,
                    'Job Description': job_description,
                    'Skills': skills,
                    'Responsibilities': responsibilities,
                    'Company': company
                }
                
                # Append the new record to the DataFrame
                global df
                df = df.append(new_record, ignore_index=True)
                
                # Save the DataFrame to the CSV file
                df.to_csv(data_path, index=False)
                
                st.success("Changes saved and new record added!")
        else:
            st.write("No matching job found.")

# Streamlit app with multiple pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Resume Job Matcher", "Job Details Search and Edit"])

if page == "Resume Job Matcher":
    page1()
else:
    page2()
