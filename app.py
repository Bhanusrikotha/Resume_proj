import streamlit as st
import PyPDF2
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="🚀 AI Resume Ranker", layout="wide")

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# =========================
# SKILLS DATABASE
# =========================
SKILLS_DB = [
    "python", "java", "c++", "sql", "aws", "azure", "gcp",
    "docker", "kubernetes", "react", "node", "express",
    "tensorflow", "pytorch", "machine learning", "deep learning",
    "nlp", "data analysis", "devops", "linux", "git"
]

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text.lower()
    except:
        return ""

# =========================
# TEXT CHUNKING
# =========================
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# =========================
# SKILL EXTRACTION
# =========================
def extract_skills(text):
    found = []
    for skill in SKILLS_DB:
        if skill in text:
            found.append(skill)
    return list(set(found))

# =========================
# EXPERIENCE EXTRACTION
# =========================
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text)
    if matches:
        return max([int(m[0]) for m in matches])
    return 0

# =========================
# SKILL MATCH SCORE
# =========================
def skill_match_score(jd_skills, resume_skills):
    if not jd_skills:
        return 0
    matched = set(jd_skills).intersection(set(resume_skills))
    return len(matched) / len(jd_skills)

# =========================
# EXPERIENCE SCORE
# =========================
def experience_score(candidate_exp, required_exp):
    if required_exp == 0:
        return 1
    return min(candidate_exp / required_exp, 1)

# =========================
# UI
# =========================
st.title("🚀 Industry-Level AI Resume Ranker")

job_description = st.text_area("📌 Enter Job Description")

required_exp = st.slider("📊 Required Experience (Years)", 0, 15, 2)

uploaded_files = st.file_uploader(
    "📄 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True
)

# =========================
# MAIN PROCESS
# =========================
if st.button("🔍 Analyze Resumes"):

    if not uploaded_files or not job_description:
        st.warning("⚠️ Please upload resumes and enter job description")
        st.stop()

    with st.spinner("⏳ Processing resumes..."):

        # =========================
        # JD PROCESSING
        # =========================
        jd_embedding = model.encode([job_description])[0]
        jd_embedding = np.array([jd_embedding]).astype('float32')
        faiss.normalize_L2(jd_embedding)

        jd_skills = extract_skills(job_description.lower())

        # =========================
        # RESUME PROCESSING
        # =========================
        resume_embeddings = []
        resume_data = []

        for file in uploaded_files:

            text = extract_text(file)

            if not text.strip():
                continue

            # Structured data extraction
            skills = extract_skills(text)
            exp = extract_experience(text)

            # Semantic embedding
            chunks = chunk_text(text)
            chunk_embeds = model.encode(chunks)
            resume_vector = np.mean(chunk_embeds, axis=0)

            resume_embeddings.append(resume_vector)

            resume_data.append({
                "name": file.name,
                "skills": skills,
                "experience": exp
            })

        # Handle no valid resumes
        if len(resume_embeddings) == 0:
            st.error("❌ No valid resume text found")
            st.stop()

        # Convert to numpy
        resume_embeddings = np.array(resume_embeddings).astype('float32')
        faiss.normalize_L2(resume_embeddings)

        # =========================
        # FAISS INDEX
        # =========================
        dimension = resume_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(resume_embeddings)

        # =========================
        # SEARCH
        # =========================
        k = min(10, len(resume_embeddings))
        distances, indices = index.search(jd_embedding, k)

        # =========================
        # SCORING
        # =========================
        results = []

        for i, idx in enumerate(indices[0]):

            data = resume_data[idx]

            semantic = float(distances[0][i])
            skill_score = skill_match_score(jd_skills, data["skills"])
            exp_score = experience_score(data["experience"], required_exp)

            # FINAL SCORE
            final_score = (
                0.4 * semantic +
                0.3 * skill_score +
                0.2 * exp_score +
                0.1 * (1 if data["experience"] > 0 else 0)
            )

            results.append({
                "name": data["name"],
                "final": final_score,
                "semantic": semantic,
                "skill": skill_score,
                "exp_score": exp_score,
                "experience": data["experience"],
                "skills": data["skills"]
            })

        # Sort results
        results = sorted(results, key=lambda x: x["final"], reverse=True)

        # =========================
        # DISPLAY RESULTS
        # =========================
        st.success("🎯 Top Matching Candidates")

        for rank, res in enumerate(results):

            st.markdown(f"""
### 🏆 Rank {rank+1}: {res['name']}

**Final Score:** {res['final']:.4f}  
- 🧠 Semantic Score: {res['semantic']:.4f}  
- 🧩 Skill Match: {res['skill']:.4f}  
- 📈 Experience Score: {res['exp_score']:.4f}  
- ⏳ Extracted Experience: {res['experience']} years  

**💡 Skills Found:** {", ".join(res['skills']) if res['skills'] else "None"}
""")
