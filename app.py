import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #0a1e3f;
}
h1 {
    color: white !important;
    font-family: 'Trebuchet MS', sans-serif;
    font-weight: 700;
}
.stCaption, p, label, .stMarkdown {
    color: white !important;
}
.stTextInput input {
    border-radius: 8px;
    padding: 10px;
    background-color: white;
    color: black;
}
.stFileUploader {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
}
div.stButton > button {
    border-radius: 8px;
    background-color: white;
    color: #0a1e3f;
    font-weight: 600;
}
.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 Chat with your PDF")
st.caption("Upload a PDF and ask questions about the content and get the answers")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
client = Groq(api_key=os.getenv("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"))  # Ensure you have your Groq API key in the .env file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "chunks" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(text)

            embeddings = model.encode(chunks)
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["file_name"] = uploaded_file.name

    st.divider()

    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("Thinking..."):
            question_embedding = model.encode([question])
            k = 3
            distances, indices = st.session_state["index"].search(np.array(question_embedding), k)
            context = "\n\n".join([st.session_state["chunks"][i] for i in indices[0]])

            prompt = f"""Answer the question using only the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
"""

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}]
            )

        st.write("**Answer:**")
        st.info(response.choices[0].message.content)