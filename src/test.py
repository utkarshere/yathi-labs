from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()


query = "What are neural networks?"

test_docs = ["Sun rises in the east.",
            "Neural Networks are dense network of neurons and edges which help adjust the parameter/weights for effective training.",
            "Machine Learning is a subclass of AI.",
            "Scikit learn is a python library."]


embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
embedded_query = embed_model.embed_query(query)
embedded_docs = embed_model.embed_documents(test_docs)

embedding_result = cosine_similarity(np.array(embedded_query).reshape(1,-1), np.array(embedded_docs))

highest_matched = np.argmax(np.array(embedding_result[0]))

print(test_docs[highest_matched])

