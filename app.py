import streamlit as st
import PyPDF2
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="🚀 AI Resume Ranker (Industry Level)", layout="wide")

# =========================
# STOPWORDS (SMART)
# =========================
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS - {
    "with", "using", "based", "via", "over", "under"
}

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

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
# CLEAN TEXT + STOPWORD REMOVAL
# =========================
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    words = text.split()

    filtered_words = [
        w for w in words
        if w not in CUSTOM_STOPWORDS and len(w) > 2
    ]

    return " ".join(filtered_words)

# =========================
# SMART PHRASE EXTRACTION
# =========================
def get_phrases(text):
    words = text.split()
    phrases = []

    for i in range(len(words)):
        w1 = words[i]

        # Single word
        if len(w1) > 2:
            phrases.append(w1)

        # Bi-gram
        if i < len(words) - 1:
            w2 = words[i + 1]
            if len(w2) > 2:
                phrases.append(f"{w1} {w2}")

        # Tri-gram
        if i < len(words) - 2:
            w2 = words[i + 1]
            w3 = words[i + 2]
            if len(w2) > 2 and len(w3) > 2:
                phrases.append(f"{w1} {w2} {w3}")

    # Remove generic phrases
    blacklist = {
        "worked on", "responsible for", "involved in",
        "project using", "project based", "good knowledge",
        "ability to", "team player", "hard working"
    }

    phrases = [p for p in phrases if p not in blacklist]

    return list(set(phrases))

# =========================
# AI SKILL EXTRACTION (UPGRADED)
# =========================
def extract_skills_ai(text, top_k=15):

    text = clean_text(text)
    phrases = get_phrases(text)

    if not phrases:
        return []

    embeddings = model.encode(phrases)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)

    # Frequency scoring
    freq = {}
    for p in phrases:
        freq[p] = freq.get(p, 0) + 1

    scores = []
    max_freq = max(freq.values())

    for i in range(len(embeddings)):
        sim = np.dot(embeddings, embeddings[i]).mean()

        # Hybrid score
        score = (0.7 * sim) + (0.3 * (freq[phrases[i]] / max_freq))
        scores.append(score)

    top_indices = np.argsort(scores)[-top_k:]
    skills = [phrases[i] for i in top_indices]

    return skills

# =========================
# EXPERIENCE EXTRACTION
# =========================
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text)
    if matches:
        return max([int(m[0]) for m in matches])
    return 0

# =========================
# SKILL MATCH USING EMBEDDINGS
# =========================
def skill_match_score(jd_skills, resume_skills):

    if not jd_skills or not resume_skills:
        return 0

    jd_emb = model.encode(jd_skills)
    res_emb = model.encode(resume_skills)

    jd_emb = np.array(jd_emb).astype('float32')
    res_emb = np.array(res_emb).astype('float32')

    faiss.normalize_L2(jd_emb)
    faiss.normalize_L2(res_emb)

    scores = np.dot(jd_emb, res_emb.T)

    return np.mean(np.max(scores, axis=1))

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
st.title("🚀 AI Resume Ranker (Industry-Level AI)")

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

        # JD processing
        jd_embedding = model.encode([job_description])[0]
        jd_embedding = np.array([jd_embedding]).astype('float32')
        faiss.normalize_L2(jd_embedding)

        jd_skills = extract_skills_ai(job_description)

        resume_embeddings = []
        resume_data = []

        for file in uploaded_files:

            text = extract_text(file)

            if not text.strip():
                continue

            skills = extract_skills_ai(text)
            exp = extract_experience(text)

            words = text.split()
            chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 300)]
            chunk_embeds = model.encode(chunks)
            resume_vector = np.mean(chunk_embeds, axis=0)

            resume_embeddings.append(resume_vector)

            resume_data.append({
                "name": file.name,
                "skills": skills,
                "experience": exp
            })

        resume_embeddings = np.array(resume_embeddings).astype('float32')
        faiss.normalize_L2(resume_embeddings)

        index = faiss.IndexFlatIP(resume_embeddings.shape[1])
        index.add(resume_embeddings)

        k = min(10, len(resume_embeddings))
        distances, indices = index.search(jd_embedding, k)

        results = []

        for i, idx in enumerate(indices[0]):

            data = resume_data[idx]

            semantic = float(distances[0][i])
            skill_score = skill_match_score(jd_skills, data["skills"])
            exp_score = experience_score(data["experience"], required_exp)

            final_score = (
                0.4 * semantic +
                0.35 * skill_score +
                0.2 * exp_score +
                0.05 * (1 if data["experience"] > 0 else 0)
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

        results = sorted(results, key=lambda x: x["final"], reverse=True)

        # Display
        st.success("🎯 Top Matching Candidates")

        for rank, res in enumerate(results):

            st.markdown(f"""
### 🏆 Rank {rank+1}: {res['name']}

**Final Score:** {res['final']:.4f}  
- 🧠 Semantic Score: {res['semantic']:.4f}  
- 🧩 AI Skill Match: {res['skill']:.4f}  
- 📈 Experience Score: {res['exp_score']:.4f}  
- ⏳ Experience: {res['experience']} years  

**💡 Extracted Skills (AI):** {", ".join(res['skills'])}
""")
