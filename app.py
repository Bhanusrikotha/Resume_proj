import streamlit as st
import PyPDF2
import numpy as np
import faiss   # 🔥 NEW (FAISS)
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="FAISS Resume Ranker", page_icon="📄", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# =========================
# EXTRACT TEXT
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
# UI
# =========================
st.title("📄 AI Resume Ranker (FAISS Powered)")
job_description = st.text_area("Enter Job Description")
uploaded_files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)

# =========================
# MAIN LOGIC
# =========================
if st.button("Analyze with FAISS"):

    if not uploaded_files or not job_description:
        st.warning("Upload resumes and enter job description")
        st.stop()

    with st.spinner("Processing..."):

        # =========================
        # STEP 1: Convert JD to vector
        # =========================
        jd_embedding = model.encode([job_description])[0]
        jd_embedding = np.array([jd_embedding]).astype('float32')
        faiss.normalize_L2(jd_embedding)  # Normalize for Cosine Similarity

        # =========================
        # STEP 2: Process resumes
        # =========================
        resume_embeddings = []
        resume_names = []

        for file in uploaded_files:

            text = extract_text(file)

            if not text.strip():
                continue

            chunks = chunk_text(text)

            # Convert chunks → embeddings
            chunk_embeds = model.encode(chunks)

            # Average all chunks → single vector
            resume_vector = np.mean(chunk_embeds, axis=0)

            resume_embeddings.append(resume_vector)
            resume_names.append(file.name)

        # Convert to numpy
        resume_embeddings = np.array(resume_embeddings).astype('float32')
        faiss.normalize_L2(resume_embeddings)  # Normalize for Cosine Similarity

        # =========================
        # STEP 3: CREATE FAISS INDEX
        # =========================
        dimension = resume_embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine Similarity

        # Add all resume vectors into FAISS
        index.add(resume_embeddings)

        # =========================
        # STEP 4: SEARCH (MAGIC)
        # =========================
        k = min(10, len(resume_embeddings))

        distances, indices = index.search(jd_embedding, k)

        # =========================
        # STEP 5: SHOW RESULTS
        # =========================
        st.success("✅ Done! Top Matches:")

        for rank, idx in enumerate(indices[0]):
            name = resume_names[idx]
            score = distances[0][rank]

            st.write(f"{rank+1}. {name} → Score: {score:.4f}")
