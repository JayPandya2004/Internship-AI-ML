import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:5000"  

st.title("Resume vs Job Description Matcher")

uploaded_files = st.file_uploader("Upload Resumes (TXT files)", accept_multiple_files=True, type=['txt'])
job_description = st.text_area("Enter Job Description")

method = st.selectbox("Choose similarity method", ['tfidf', 'word2vec', 'bert'])

if st.button("Compare"):
    if not uploaded_files:
        st.error("Please upload at least one resume.")
    elif not job_description.strip():
        st.error("Please enter a job description.")
    else:
        # Read resumes text
        resumes = [file.read().decode('utf-8') for file in uploaded_files]

        # Call backend compare API
        payload = {
            "job_description": job_description,
            "resumes": resumes,
            "method": method
        }

        try:
            response = requests.post(f"{BACKEND_URL}/compare", json=payload)
            response.raise_for_status()
            data = response.json()
            rankings = data.get("rankings", [])

            st.subheader("Similarity Rankings:")
            for rank in rankings:
                idx = rank['resume_index']
                score = rank['similarity']
                st.write(f"Resume {idx + 1}: Similarity Score = {score:.4f}")

        except Exception as e:
            st.error(f"Error calling backend API: {e}")
