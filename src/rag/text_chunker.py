from rag.doc_loader import document_loader

documents = document_loader()

if documents:
    print(f"\nTotal documents loaded: {len(documents)}")

else:
    print("\nNo documents were loaded.")
    

doc = documents[3]

print(doc.get_content())