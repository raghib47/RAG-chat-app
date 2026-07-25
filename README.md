# 📄 RAG PDF Chat

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask questions grounded in their content — built with **LangChain**, **Mistral AI**, **Chroma**, and **Streamlit**.

## Features

- 📤 Upload one or more PDFs directly from the browser
- ✂️ Automatic chunking (`RecursiveCharacterTextSplitter`) and embedding (`MistralAIEmbeddings`)
- 🔍 MMR-based retrieval from a Chroma vector store for diverse, relevant context
- 💬 Chat interface with full conversation history
- 📎 Expandable source panel showing which document/page each answer came from
- 🎨 Dark-themed, custom-styled UI

## Project structure

```
rag-pdf-chat/
├── app.py               # Streamlit app (upload, index, chat)
├── requirements.txt
├── .env.example
├── .gitignore
└── legacy/              # Original standalone CLI scripts
    ├── create_database.py
    └── main.py
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/rag-pdf-chat.git
   cd rag-pdf-chat
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add your Mistral API key**
   ```bash
   cp .env.example .env
   # then edit .env and paste your key
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the local URL Streamlit prints (usually `http://localhost:8501`), upload a PDF in the sidebar, click **Process**, and start chatting.

## Tech stack

- [Streamlit](https://streamlit.io/) — UI
- [LangChain](https://www.langchain.com/) — orchestration
- [Mistral AI](https://mistral.ai/) — embeddings + chat model (`mistral-small-2603`)
- [Chroma](https://www.trychroma.com/) — vector store

## Notes

- Each session builds its vector store in a temporary directory, so re-processing documents won't overwrite a shared `chroma_db` folder.
- Retrieval uses MMR (`k=4`, `fetch_k=10`, `lambda_mult=0.5`) to balance relevance and diversity in retrieved chunks.

## License

MIT — see [LICENSE](LICENSE).

