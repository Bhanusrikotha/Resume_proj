import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Hybrid ATS Resume Ranker", page_icon="📄", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# =========================
# EXTRACT TEXT FROM PDF
# =========================
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# =========================
# CHUNK TEXT
# =========================
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# =========================
# KEYWORD EXTRACTION
# =========================
def extract_keywords(text):
    words = text.lower().split()
    return set(words)

# =========================
# KEYWORD MATCH SCORE
# =========================
def keyword_match_score(jd_text, resume_text):
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)

    if len(jd_keywords) == 0:
        return 0

    matched = jd_keywords.intersection(resume_keywords)
    return len(matched) / len(jd_keywords)

# =========================
# UI
# =========================
st.title("📄 Hybrid AI Resume Ranker (Semantic + Keyword)")

job_description = st.text_area("Enter Job Description")
uploaded_files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)

# =========================
# MAIN LOGIC
# =========================
if st.button("Analyze Resumes"):

    if not uploaded_files or not job_description:
        st.warning("Upload resumes and enter job description")
        st.stop()

    with st.spinner("Processing..."):

        # =========================
        # STEP 1: JD EMBEDDING
        # =========================
        jd_embedding = model.encode([job_description])[0]
        jd_embedding = np.array([jd_embedding]).astype('float32')
        faiss.normalize_L2(jd_embedding)

        # =========================
        # STEP 2: PROCESS RESUMES
        # =========================
        resume_embeddings = []
        resume_names = []
        resume_texts = []

        for file in uploaded_files:

            text = extract_text(file)

            if not text.strip():
                continue

            resume_texts.append(text)
            resume_names.append(file.name)

            chunks = chunk_text(text)

            chunk_embeds = model.encode(chunks)

            # Average chunk embeddings
            resume_vector = np.mean(chunk_embeds, axis=0)

            resume_embeddings.append(resume_vector)

        # Convert to numpy
        resume_embeddings = np.array(resume_embeddings).astype('float32')
        faiss.normalize_L2(resume_embeddings)

        # =========================
        # STEP 3: FAISS INDEX
        # =========================
        dimension = resume_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(resume_embeddings)

        # =========================
        # STEP 4: SEARCH
        # =========================
        k = min(10, len(resume_embeddings))
        distances, indices = index.search(jd_embedding, k)

        # =========================
        # STEP 5: HYBRID SCORING
        # =========================
        results = []

        for i, idx in enumerate(indices[0]):

            name = resume_names[idx]
            semantic_score = distances[0][i]
            resume_text = resume_texts[idx]

            # Keyword score
            keyword_score = keyword_match_score(job_description, resume_text)

            # Final hybrid score
            final_score = (0.6 * semantic_score) + (0.4 * keyword_score)

            results.append((name, final_score, semantic_score, keyword_score))

        # Sort by final score
        results.sort(key=lambda x: x[1], reverse=True)

        # =========================
        # DISPLAY RESULTS
        # =========================
        st.success("✅ Top Matching Resumes:")

        for rank, (name, final, sem, key) in enumerate(results):
            st.write(f"""
            🔹 **Rank {rank+1}: {name}**
            - Final Score: {final:.4f}
            - Semantic Score: {sem:.4f}
            - Keyword Score: {key:.4f}
            """)
