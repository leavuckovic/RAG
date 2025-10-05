from flask import Blueprint, request, jsonify, current_app
import os, uuid, logging
from app.utils import  list_files_in_folder, sanitize_entities
from app.qa_model import response, load_documents, vector_database, ocr_image_to_text, extract_entities
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

load_dotenv()  
logging.basicConfig(
    level=logging.INFO,  # or INFO, WARNING, ERROR
    format='%(asctime)s - %(levelname)s - %(message)s'
)
api = Blueprint('api', __name__)

#Uncomment if you want to use COHERE ,  LIMITED!!!
# cohere_api_key = os.getenv("COHERE_API_KEY")
# llm = ChatCohere(model="command-r-plus-08-2024", temperature=0, cohere_api_key=cohere_api_key)
# embeddings= CohereEmbeddings(model="embed-english-v3.0",cohere_api_key=cohere_api_key)

llm = OllamaLLM(model='gemma:2b', temperature=0) #Comment if you are using COHERE LLM
embeddings = OllamaEmbeddings(model='nomic-embed-text') #Comment if you are using COHERE embedding


@api.route('/upload', methods=['POST'])
def upload():
    """Uploading the documents."""

    files = request.files.getlist('documents')
    session_id = request.form.get('session_id', str(uuid.uuid4()))
    session_path = os.path.join('data', session_id)
    os.makedirs(session_path, exist_ok=True)

    # Save uploaded files
    for file in files:
        filepath = os.path.join(session_path, file.filename)
        file.save(filepath)
    logging.info(f"Uploaded documents saved in folder {session_path}!")
    
    # Load and split documents
    documents = []
    for file in files:
        filepath = os.path.join(session_path, file.filename)
        ext = os.path.splitext(file.filename)[1].lower()

        if ext in ['.pdf', '.txt']:
            docs = load_documents(filepath)
            documents.extend(docs)
        elif ext in ['.png', '.jpg', '.jpeg']:
            doc = ocr_image_to_text(filepath)
            documents.extend(doc)

    # Create retriever and store it per session
    retriever = vector_database(documents, embeddings, num_docs=3)
    current_app.config.setdefault("retrievers", {})[session_id] = retriever

    #List the documents
    files_list = list_files_in_folder(session_path)
    logging.info(f"Retriever is created from documents {files_list}!")

    return jsonify({'session_id': session_id, 'message': 'Files uploaded and indexed.'})



@api.route('/ask', methods=['POST'])
def ask():
    """Answering the user question."""

    data = request.json
    session_id = data['session_id']
    question = data['question']
    logging.info(f"Question: '{question}'")

    # Retrieve retriever from session
    retriever = current_app.config.get("retrievers", {}).get(session_id)
    if not retriever:
        return jsonify({'error': 'Session not found or retriever not initialized.'}), 400

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)
    logging.info("Chunks retrieved!")
    for id,doc in enumerate(retrieved_docs):
        print("CHUNK ID ",id)
        print("CHUNK CONTENT ",doc.page_content)
        print("---------------------------")
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Run QA
    answer = response(question, context)
    logging.info(f"Answer is {answer}.")
    entities = extract_entities(answer)
    logging.info(f"NER analysis is {entities}")

    return jsonify({'answer': answer,'entities':sanitize_entities(entities)})





