import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    txt_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
    all_files = txt_files + pdf_files + csv_files
    return [load_single_document(file_path) for file_path in all_files]


class DocumentLoader:
    def __init__(self):
        llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
        model_n_ctx = os.environ.get('MODEL_N_CTX')
        self.persist_directory = os.environ.get('PERSIST_DIRECTORY')
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)

    def load_doc_content(self, doc_name: str, content: str):
        document = Document(page_content=content, metadata={"source": doc_name})
        texts = self.text_splitter.split_documents([document])
        db = Chroma.from_documents(
            texts,
            self.llama,
            persist_directory=self.persist_directory,
            client_settings=CHROMA_SETTINGS
        )
        db.persist()
