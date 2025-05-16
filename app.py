import os
import gradio as gr
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Caricamento PDF
documents = []
data_path = "data"
os.makedirs(data_path, exist_ok=True)
for file in os.listdir(data_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(data_path, file))
        documents.extend(loader.load())

# Split del testo
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Embedding + FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_model)

# Pipeline Hugging Face (LLM)
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Creazione RAG chain con RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Funzione da collegare a Gradio
def answer_question(query):
    return qa_chain.run(query)

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
