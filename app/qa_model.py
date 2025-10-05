from transformers import pipeline
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain.docstore.document import Document
import easyocr
import os
from dotenv import load_dotenv

load_dotenv()  # Loads from .env file

#Uncomment if you want to use COHERE ,  LIMITED!!!
# cohere_api_key = os.getenv("COHERE_API_KEY")
# llm = ChatCohere(model="command-r-plus-08-2024", temperature=0, cohere_api_key=cohere_api_key)
# embeddings= CohereEmbeddings(model="embed-english-v3.0",cohere_api_key=cohere_api_key)



#LLM 
llm = OllamaLLM(model='gemma:2b', temperature=0) #Comment if you are using COHERE LLM
#Embedding model
embeddings = OllamaEmbeddings(model='nomic-embed-text') #Comment if you are using COHERE embedding
#NER    
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
#OCR reader
reader = easyocr.Reader(['en'])  



def load_documents(input_path):
    """ Loading the documents from folder and  splitting the document in chunks"""
    
    loader = PyMuPDFLoader(input_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=512,
    )
    chunks = text_splitter.split_documents(pages)
    return chunks



def ocr_image_to_text(image_path):
    """Reading text from the image"""
    results = reader.readtext(image_path, detail=0)
    text = "\n".join(results)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=512,
    )
    doc=Document(page_content=text, metadata={"source": image_path})
    chunks = text_splitter.split_documents([doc])
    return chunks





def vector_database(documents, embeddings, num_docs):
    """"Creating vector database, embedding the chunks and storing in vector database"""
    
    d = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(d)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_documents(documents=documents)
    retriever = vector_store.as_retriever(search_kwargs={"k": num_docs})
    return retriever





def response(question,context):
    """Generating an answer based on retrieved context"""
    
    input_data = {
        "question": question,
    }

    system_prompt = """
        You are a helpful assistant designed to answer questions using only the provided context. 
        Your task is to extract accurate information strictly from the given content, which may include text, tables, or structured data.

        - Think step by step before answering.
        - Do not use any external knowledge or assumptions.
        - If the answer is clearly stated in the context, respond with a concise and complete answer.
        - If the answer is not present or cannot be determined, respond with: "I am not sure."
        - Do not explain your reasoning or repeat the question.
        - Do not mention the context or your limitations.

        Respond only with the final answer.
        """
    
    template = system_prompt + """
    CONTEXT:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("human", "{question}")
        ])

    chain = (
        RunnablePassthrough.assign(context=lambda x: context)
        | prompt
        | llm 
        | StrOutputParser()
    )

    result = chain.invoke(input_data)
    return result


def extract_entities(text):
    """Extract the entities from the answer"""
    entities = ner(text)
    return entities




