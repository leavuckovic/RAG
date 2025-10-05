#  Retrieval-Augmented Generation (RAG) App

This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents and ask questions about them. It combines the power of modern LLMs with custom embeddings to deliver accurate, context-aware answers.

## Features Overview
###  Document & Image Upload
- Supported formats: PDF, JPG, PNG, JPEG
- PDF Parsing: Utilizes PyMuPDF for fast and reliable text extraction
- Image OCR: Powered by EasyOCR to extract readable text from scanned images or photos

###   Interactive Chat Interface
- Built with Streamlit for a responsive and intuitive user experience
- Users can ask natural language questions directly about uploaded content
- Real-time responses generated using integrated LLMs

###   Retrieval-Augmented Generation (RAG) Pipeline
- Chunking: Uploaded documents are split into manageable text chunks
- Embedding: Each chunk is embedded using language model embeddings
- Vector Storage: Embeddings are stored in FAISS, a high-performance vector database
- Retrieval: Relevant chunks are retrieved based on user queries
- Generation: Retrieved context is passed to the LLM for grounded answer generation
- Execution: Orchestrated using LangChain for modular and scalable chain execution

###   Language Model Integration
- Supports Ollama and Cohere for flexible and powerful text generation
- Easily extendable to other LLM providers

###   Named Entity Recognition (NER)
- Integrated transformers model: dslim/bert-base-NER
- Automatically identifies and highlights key entities (names, organizations, dates, etc.) within documents

##  Architecture
###   Backend
- Built with Flask to handle file uploads, preprocessing, and API endpoints
- Manages RAG pipeline, NER tagging, and LLM orchestration

###   Frontend
- Developed using Streamlit
- Provides a clean, interactive interface for document upload and chat-based querying


## Modular components
| **Component**       | **Technology**             | **Purpose**                                      |
|---------------------|----------------------------|--------------------------------------------------|
| Document Parsing     | PyMuPDF, EasyOCR           | Extract text from PDFs and images                |
| Embedding & RAG      | LangChain, FAISS           | Retrieve relevant context for generation         |
| LLMs                 | Ollama, Cohere             | Generate natural language answers                |
| NER                  | transformers               | Extract named entities from text                 |
| Backend              | Flask                      | API and pipeline orchestration                   |
| Frontend             | Streamlit                  | User interface for chat and uploads              |


## Project Structure

```text
RAG/
├── app/
│   ├── __init__.py
│   ├── qa_models.py
│   ├── routes.py
│   └── utils.py
├── data/
├── dummy_doc/
├── venv
├── .gitignore
├── example.env
├── README.md
├── requirements.txt
├── run.py
└── streamlit_app.py
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/leavuckovic/RAG.git
   cd RAG

2. **Install Ollama** 
Follow instructions at [ollama.com](https://ollama.com/) to install and run Ollama locally.

4. **Pull models from Ollama**
- LLM model:
```bash
git pull gemma:2b
```
- Embedding model:
```bash
git pull nomic-embed-text
```
5. **Get COHERE trial API key (Optional)**
- Go to COHERE page (https://dashboard.cohere.com/) and get API key
- Set API key in .env file

6. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

5. **Run the app:**
Open bash terminal and run:
- Flask (Backend)
```bash
python run.py
```
- Streamlit (Frontend)
```bash
streamlit run streamlit_app.py
```

## Example Questions:
### Input: image - invoice_image.png
- Tell me property details.  
Answer: Sure, here are the property details from the context:  Type: Famiiy House  Unit Price (EUR): 250,000,00  Total (EUR): 250,000,00  

- How much is advance payment?  
Answer: The advance payment is 25,000,00.

- What is invoice number?  
Answer: The invoice number is RE-2025-001.

- What is the date of the issue?  
Answer: The date of issue is October 5, 2025.
### Input: documents : Contract.pdf
- Who is buyer?  
Answer: The buyer is Ana Kovač.

- Who is seller?  
Answer: The seller is Marko Horvat.

- What is the address and purchase price on house?  
Answer: Sure, here is the answer  
Address: Braće Stipčić Street 12, 51000 Rijeka Purchase Price: EUR 250,000.00

- Is there tax obligation?   
Answer: Yes, the buyer is obligated to pay real estate transfer tax in the amount of 3% of the purchase price, which is EUR 25,000.00.

- How many days in advance buyer needs to pay advance payment?  
Answer: According to the context, the buyer needs to pay an advance payment of EUR 25,000.00 within 5 days of signing the agreement.

