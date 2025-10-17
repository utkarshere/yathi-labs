import os





doc_path = os.get('./data/document.pdf')



def chunk_doc(doc):
    """Effectively """

    


    
def embeddeddoc(chunked_doc):
    embeddings = []
    for chunk in chunked_doc:
        embedding = OpenAIembedding.embed_query(chunk)
    embeddings.append(embedding)
    return embeddings
