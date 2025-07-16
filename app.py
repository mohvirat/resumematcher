
# Resume Match and Ranking Script - Streamlit Deployable Version with Login

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import difflib

# Set OpenAI API Key (ensure it's set in your environment or secrets.toml)
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Missing OpenAI API Key. Please set OPENAI_API_KEY in environment variables or Streamlit secrets.")
openai.api_key = api_key

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def calculate_similarity(text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([jd_text, text])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

def generate_ai_summary(text):
    prompt = f"""
    Analyze the following resume content:

    {text[:4000]}

    Provide a summary with:
    - Key strengths
    - Weaknesses
    - Technologies used
    - Project experience
    Keep it clear and concise.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional technical recruiter."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"AI Summary failed: {e}"

def fuzzy_match(term, text, threshold=0.6):
    words = text.split()
    matches = [word for word in words if difflib.SequenceMatcher(None, word.lower(), term.lower()).ratio() >= threshold]
    return bool(matches)

def evaluate_skills(resume_text, essential_skills, preferred_skills):
    all_skills = essential_skills + preferred_skills
    resume_lower = resume_text.lower()
    skill_results = {}
    for skill in all_skills:
        match_found = fuzzy_match(skill, resume_lower)
        skill_results[skill] = "🟢" if match_found else "🔴"
    essential_match = sum(skill_results[k] == "🟢" for k in essential_skills)
    preferred_match = sum(skill_results[k] == "🟢" for k in preferred_skills)

    essential_score = (essential_match / len(essential_skills)) * 70 if essential_skills else 0
    preferred_score = (preferred_match / len(preferred_skills)) * 30 if preferred_skills else 0
    weighted_score = round(essential_score + preferred_score, 2)

    return skill_results, weighted_score, essential_match, preferred_match

def color_match_level(score):
    if score >= 80:
        return "🟢 High"
    elif score >= 60:
        return "🟡 Medium"
    else:
        return "🔴 Low"

# ---------------------------
# STREAMLIT APP
# ---------------------------

def main():
    st.set_page_config(page_title="Resume Matcher", layout="centered")
    st.title("🔐 Secure Resume Matcher")

    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login = st.form_submit_button("Login")

    if login:
        if username == "Virat" and password == "KleisTech@123":
            st.success("✅ Access granted. Welcome, Virat!")

            with st.sidebar:
                st.markdown("### Step 1: Job Description")
                jd_text = st.text_area("Paste the Job Description here:")

                st.markdown("### Step 2: Required Skills")
                essential_input = st.text_area("Essential Skills (comma-separated)", "Python, SQL")
                preferred_input = st.text_area("Preferred Skills (comma-separated)", "Docker, Kubernetes")

                st.markdown("### Step 3: Upload Resumes")
                resume_files = st.file_uploader("Upload PDF resumes", type="pdf", accept_multiple_files=True)

            if jd_text and resume_files:
                essential_skills = [s.strip() for s in essential_input.split(',') if s.strip()]
                preferred_skills = [s.strip() for s in preferred_input.split(',') if s.strip()]

                results = []

                with st.spinner("Processing resumes..."):
                    for resume_file in resume_files:
                        try:
                            text = extract_text_from_pdf(resume_file)
                            ai_summary = generate_ai_summary(text)
                            skill_map, weighted_score, essential_hit, preferred_hit = evaluate_skills(
                                text, essential_skills, preferred_skills)

                            results.append({
                                "Candidate": resume_file.name,
                                "Skill Match %": weighted_score,
                                "Match Level": color_match_level(weighted_score),
                                "Essential Skills": f"{essential_hit}/{len(essential_skills)}",
                                "Preferred Skills": f"{preferred_hit}/{len(preferred_skills)}",
                                "Skills Table": skill_map,
                                "AI Summary": ai_summary
                            })
                        except Exception as e:
                            st.error(f"Error processing {resume_file.name}: {e}")

                if results:
                    df = pd.DataFrame(results)
                    st.subheader("📊 Summary Table")
                    st.dataframe(df[["Candidate", "Skill Match %", "Essential Skills", "Preferred Skills", "Match Level"]].sort_values(by="Skill Match %", ascending=False))

                    csv = df.drop(columns=["Skills Table", "AI Summary"]).to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download CSV", csv, "resume_match_results.csv", "text/csv")

                    st.subheader("🧠 Skills Table")
                    for res in results:
                        st.markdown(f"### {res['Candidate']}")
                        skill_df = pd.DataFrame(list(res['Skills Table'].items()), columns=["Skill", "Match"])
                        st.dataframe(skill_df)

                    st.subheader("🤖 AI Summaries")
                    for res in results:
                        st.markdown(f"### {res['Candidate']}")
                        st.markdown(res['AI Summary'])
        else:
            st.error("❌ Invalid username or password.")

if __name__ == "__main__":
    main()
