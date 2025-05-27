import streamlit as st
import fitz  # PyMuPDF
import docx
from keybert import KeyBERT
from transformers import pipeline

# Initialize models
kw_model = KeyBERT()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def extract_text(file):
    if file.name.endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        return text
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return "Unsupported file format"

def classify_text(text):
    labels = ["Resume", "Invoice", "Report", "Article", "Letter", "Assignment"]
    result = classifier(text[:1000], candidate_labels=labels)
    return result['labels'][0]

def extract_tags(text):
    return kw_model.extract_keywords(text, top_n=5)

# Streamlit UI
st.title("📁 Smart File Tagger")

file = st.file_uploader("Upload your document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

if file:
    with st.spinner("Reading file..."):
        content = extract_text(file)
        st.subheader("📄 Extracted Text")
        # Show the first 2000 characters or full content if shorter
        st.text_area("Document Text", content[:2000] + "..." if len(content) > 2000 else content, height=300)

    with st.spinner("Classifying document..."):
        doc_type = classify_text(content)
        st.success(f"🧾 Document Type: **{doc_type}**")

    with st.spinner("Extracting tags..."):
        tags = extract_tags(content)
        st.markdown("🏷️ **Suggested Tags:**")
        st.write([tag[0] for tag in tags])
