from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIembedddings
from langchain.core import PromptTemplate
from Pydantic import BaseModel, Field
from langchain.outputparsers import JSONOutputparser
from dotenv import load_env
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.docprocess import embeddeddoc


load_env()



test_doc = ["Neural networks are dense layers of neurons and edges which optimize the loss function based on computing gradients.",
"Neural networks are trained through an algorithm called gradient descent which helps adjusts to optimized weights.",
"Neural networks are the core behind the deep learning techniques"]

embeddings = OpenAIembeddings.embed_documents(test_doc)


class Subquery(BaseModel):
    query = Field(description="simple and self contained questions about a topic")



parser = JSONOutputparser(query=Subquery)

def decomposition_query(query : Optional[str]):

    """ Decomposes a complex question into sub questions for fetching effective context."""




    

template = PromptTemplate("""You are an expert in answering user queries based on the context provided from the documents.
                          """, )










