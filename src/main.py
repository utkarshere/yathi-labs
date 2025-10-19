from src.docprocess import load_and_extract, token_chunking, embed_store

pdf_path = './data/document.pdf'
CHUNK_SIZE = 200
OVERLAP = 20

try:
    
    print("Step 1: Loading PDF...")
    print("=" * 50)
    doc_text = load_and_extract(pdf_path)
    
    if not doc_text:
        print("Failed to extract text from PDF")
        exit(1)
    
    print(f"✓ Extracted {len(doc_text)} characters\n")
    

    print("Step 2: Chunking text...")
    print("=" * 50)
    tokens = token_chunking(doc_text, CHUNK_SIZE, OVERLAP)
    
    if not tokens:
        print("Failed to create chunks")
        exit(1)
    
    print(f"✓ Created {len(tokens)} chunks\n")
    
   

    print("Step 3: Creating embeddings...")
    print("=" * 50)
    vector_store = embed_store(tokens)
    
    if not vector_store:
        print("Failed to create vector store")
        exit(1)
    
    print(f"✓ Vector store created successfully\n")
    
    
    print("=" * 50)
    print("Testing Query...")
    print("=" * 50)
    query = "What is this document about?"
    results = vector_store.similarity_search(query, k=2)
    
    print(f"\nQuery: {query}")
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content[:200] + "...")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")