from typing import Optional
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import os
from dotenv import load_dotenv

load_dotenv()


def load_and_extract(doc_path)->Optional[str]:
    """Loads a file from file path and returns extracted page text.
    Args:
        doc_path: path where original document is located.
    Returns:
        all_text: string containing complete extracted text from processed pdf pages.
        """

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"File not found: {doc_path}")
    else:
        pdf = PdfReader(doc_path)

    try:
        all_text = ""
        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text()
            all_text+= text + "\n\n"

        return all_text.strip()
    
    except PdfReadError as e:
        print(f"Unable to read pdf - {e}")
    except Exception as e:
        print(f"Error: Unexpected error occured: {e}")
    

def token_chunking(text: str, chunksize=250, overlap=25)->Optional[list[str]]:
    """Performs word/token level chunking of the text input and return list of tokens.
    Args:
        text: string containing full text.
        chunksize: size for chunk window.
        overlap: overlap length between words.
    Returns:
        list of chunks"""

    if not text:
        print("No text found to create chunks.")
        return None
    
    words= text.split()

    all_chunks= []

    for i in range(0, len(words), chunksize-overlap):
        chunk = ' '.join(words[i:i+chunksize])
        if chunk.strip():
            all_chunks.append(chunk)  

    return all_chunks  

def embed_store(
        chunks: list[str], 
        metadata: Optional[list[dict]]= None
        )-> Optional[InMemoryVectorStore]:
    """Creates embeddings from chunks and stores them in a vector store.
    
    Args:
        chunks: List of text chunks.
        metadata: list of dictionary containing associated metadata with each chunk.
    Returns:
        InMemoryVectorStore with embeddings of chunks."""
    
    if not chunks:
        print("Not chunks found to embed and store.")
        return None
    try:
    
        embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
        documents = []
        for i, chunk in enumerate(chunks):
            meta = metadata[i] if metadata and i < len(metadata) else {"chunk_id" : i}
            doc = Document(page_content=chunk, metadata=meta)
            documents.append(doc)
        
        vectorstore= InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=embed_model
        )
        return vectorstore
    except Exception as e:
        print(f"Error creating the vector store. {e}")
