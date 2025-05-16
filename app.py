import os
import gradio as gr
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Caricamento documenti PDF
documents = []
data_path = "data"
for file in os.listdir(data_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(data_path, file))
        documents.extend(loader.load())

# Split del testo
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Embedding + FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_model)

# Modello LLM gratuito Hugging Face
llm_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=200)

# RAG chain
def answer_question(query):
    context_docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""Contesto:
{context}

Domanda: {query}
Risposta:"""
    output = llm_pipeline(prompt)[0]["generated_text"]
    return output[len(prompt):].strip()

# Interfaccia Gradio
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Domanda", placeholder="Chiedimi qualcosa..."),
    outputs=gr.Textbox(label="Risposta"),
    title="Biocerto.AI Open - Chatbot Agroalimentare",
    description="Sistema RAG gratuito con modelli open-source. Inserisci i tuoi certificati nella cartella /data."
)

if __name__ == "__main__":
    demo.launch()
