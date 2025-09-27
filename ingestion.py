import os

from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    pdf_directory = r"C:\Users\lucas\OneDrive - Paul Scherrer Institut\VedaTIMESdocumentation"
    all_raw_documents = []
    #Iterate over all files in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            print(filename)
            loader = PyMuPDFLoader(file_path)
            raw_documents = loader.load()
            all_raw_documents.extend(raw_documents)
    
    #sys.exit()
    print(f"loaded {len(all_raw_documents)} documents")
        

    # send the raw documents to the text splitter, the cunk size is defined as a rule of thumb as (roughly) 2000 tokens / 4 contexts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(all_raw_documents)

   #for doc in documents:
   #    new_url = doc.metadata["source"]
   #    new_url = new_url.replace("langchain-docs", "https:/")
   #    doc.metadata.update({"source": new_url})P

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name="vedatimeshelper")
    print("****Loading to vectorstore done ***")
    


def ingest_docs2() -> None:
    from langchain_community.document_loaders.firecrawl import FireCrawlLoader

    langchain_documents_base_urls = [
        "https://python.langchain.com/docs/integrations/chat//",
        "https://python.langchain.com/docs/integrations/llms/",
        "https://python.langchain.com/docs/integrations/text_embedding/",
        "https://python.langchain.com/docs/integrations/document_loaders/",
        "https://python.langchain.com/docs/integrations/document_transformers/",
        "https://python.langchain.com/docs/integrations/vectorstores/",
        "https://python.langchain.com/docs/integrations/retrievers/",
        "https://python.langchain.com/docs/integrations/tools/",
        "https://python.langchain.com/docs/integrations/stores/",
        "https://python.langchain.com/docs/integrations/llm_caching/",
        "https://python.langchain.com/docs/integrations/graphs/",
        "https://python.langchain.com/docs/integrations/memory/",
        "https://python.langchain.com/docs/integrations/callbacks/",
        "https://python.langchain.com/docs/integrations/chat_loaders/",
        "https://python.langchain.com/docs/concepts/",
    ]

    langchain_documents_base_urls2 = [
        "https://python.langchain.com/docs/integrations/chat/"
    ]
    for url in langchain_documents_base_urls2:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="scrape",
        )
        docs = loader.load()

        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name="vedatimeshelper"
        )
        print(f"****Loading {url}* to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()