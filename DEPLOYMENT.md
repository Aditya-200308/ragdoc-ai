# 🚀 Deployment Guide: Deploying RAGDoc AI to the Web (100% Free)

This guide walks you through deploying your **RAGDoc AI** app to **Streamlit Community Cloud** so anyone can use it live on the web!

---

## 📋 Prerequisites Checklist

- [x] `requirements.txt` generated in project root
- [x] Multi-document parsing & RapidOCR enabled
- [x] Streamlit Community Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- [x] GitHub Repository with your project code

---

## 🛠️ Step 1: Push Code to GitHub

1. Open your terminal in the project directory:
   ```bash
   git init
   git add .
   git commit -m "Deploy RAGDoc AI"
   ```
2. Create a new public GitHub repository named `ragdoc-ai`.
3. Push your code:
   ```bash
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/ragdoc-ai.git
   git branch -M main
   git push -u origin main
   ```

---

## ☁️ Step 2: Deploy to Streamlit Community Cloud (Recommended)

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click **"New App"**.
3. Fill in the deployment form:
   - **Repository**: `YOUR_GITHUB_USERNAME/ragdoc-ai`
   - **Branch**: `main`
   - **Main file path**: `src/app.py`
4. Click **"Advanced settings..."** ➔ Under **Secrets**, add your free Gemini API key for online fallback:
   ```toml
   GEMINI_API_KEY = "your_free_gemini_api_key_here"
   ```
5. Click **"Deploy!"** 🚀

---

## 🌐 How The Deployed App Works Online:

- **Local Machine**: Uses **Ollama (Llama 3.2)** locally.
- **Online Web App**: Automatically routes LLM generation through **Gemini Flash (Free API)** via Cloud Secrets, while ChromaDB, PyMuPDF, and RapidOCR run embedded on the server!

---

🎉 Your live app will be accessible at:  
`https://YOUR_GITHUB_USERNAME-ragdoc-ai.streamlit.app`
