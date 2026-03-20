# Bank of Maharashtra Loan Assistant (RAG-based AI System)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based AI assistant that answers user queries related to Bank of Maharashtra loan products.

The system retrieves relevant information from a curated knowledge base (built from the official bank website) and generates accurate, context-aware responses using a Large Language Model (LLM).

The primary goal is to ensure factual correctness while minimizing hallucination by grounding responses strictly in retrieved data.

---

## Features

- Answers queries about multiple loan types:
  - Home Loans (including Maha Super Flexi Housing Loan)
  - Personal Loans
  - Education Loans
  - Gold Loans
  - Government schemes (PMAY)
- Semantic search using embeddings
- Fast similarity search using FAISS
- Context-grounded answer generation
- Explainability through retrieved context display

---

## Project Structure
bank-loan-rag-assistant/
│
├── data/
│ └── Raw Loans.txt 
│
├── preprocess.py 
├── rag_pipeline.py 
│
├── faiss_index.index 
├── chunks.npy 
│
└── README.md

---

## Project Setup

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd bank-loan-rag-assistant
```

### 2. Install dependencies

```bash
pip install faiss-cpu sentence-transformers google-genai numpy
```

### 3. Set up API Key (Google Gemini)

Run this in powershell
```bash
$env:GEMINI_API_KEY="api_key_here"
```

### 4. Run preprocessing
This step creates embeddings and builds the FAISS index.
```bash
python preprocess.py
```
Outputs:
'faiss_index.index'
'chunks.npy'

### 5. Run the assistant

```bash
python rag_pipeline.py
```
- You can now ask questions like:
- What is the interest rate for gold loan?
- What is the tenure of personal loan?
- Explain Maha Super Flexi Housing Loan
- What subsidy is available under PMAY?

---

## How It Works

1. User inputs a query
2. Query is converted into vector embedding
3. FAISS retrieves top relevant chunks from knowledge base
4. Retrieved context is passed to the LLM
5. LLM generates answer strictly based on context
This ensures factual accuracy and reduces hallucination.

---

## Architectural Decisions

### 1. RAG Architecture

A Retrieval-Augmented Generation approach was chosen to ensure responses are grounded in real data rather than relying solely on LLM knowledge.

### 2. Embedding Model

**Model:** 'all-MiniLM-L6-v2'
**Reason:**
- Lightweight and fast
- Good semantic similarity performance
- Suitable for local execution

### 3. Vector Store

**Tool:** FAISS
**Reason:**
- Efficient similarity search
- Lightweight and easy to integrate
- No need for external database

### 4. LLM Selection

**Model:** Google Gemini (via google-genai)
**Reason:**
- Free-tier availability
- Strong text generation capabilities
- Easy API integration

### 5. Data Strategy

- Source: Official Bank of Maharashtra website
- Only loan-related pages were used
- Data was manually cleaned and structured
- Sections were clearly separated for better chunking

Due to dynamic rendering of website, a hybrid approach of extraction + manual structuring was used to ensure data quality.

### 6. Chunking Strategy

- Data split using logical separators (===)
- Each chunk represents a meaningful section
- Improves retrieval accuracy

---

## Challenges Faced

### 1. Website Content Extraction

- Some pages restricted copy or had inconsistent formatting
- Solution: Manual extraction and structured cleaning

### 2. Noisy Data

- Presence of HTML tags and irrelevant text
- Solution: Cleaned data into structured plain text

### 3. Encoding Issues

- Faced UTF-8 decoding errors
- Solution: Used alternate encoding '(latin-1)' and error handling

### 4. Model API Issues

- Faced deprecation of older Gemini SDK
- Model name mismatches caused errors
Solution:
- Migrated to google-genai
- Validated available models manually from [ai.google.dev/gemini-api](https://ai.google.dev/gemini-api/docs/models)

---

## Potential Improvements

- Use a larger and continuously updated dataset

- Implement chunk overlap for better retrieval

- Add a frontend interface (Streamlit or web app)

- Improve answer formatting with structured outputs

- Add evaluation metrics (accuracy, latency)

- Deploy as a web API or cloud-based service

- Integrate conversation memory for multi-turn queries

---

## Demo

(Add your video link here)

---

## Conclusion

This project demonstrates an end-to-end implementation of a RAG-based AI system capable of delivering accurate, context-driven responses using real-world data.

---