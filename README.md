# Biocerto.AI Open

Chatbot RAG (Retrieval-Augmented Generation) per il settore agroalimentare. Utilizza modelli open-source e documenti PDF caricati nella cartella `/data`.

## Caratteristiche

- Modello LLM: `google/flan-t5-base` (gratuito, CPU-friendly)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Vector Store: FAISS
- Interfaccia: Gradio
- Sistema RAG via LangChain `RetrievalQA`

## Come usare

1. Crea una cartella `data/` nella root del progetto.
2. Inserisci i tuoi PDF certificati nella cartella `data/`.
3. Installa le dipendenze:

```
pip install -r requirements.txt
```

4. Avvia l'app:

```
python app.py
```

## Demo

Una volta avviata l'app, accedi all'interfaccia su `http://localhost:7860`.

---
