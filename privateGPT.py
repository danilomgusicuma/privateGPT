from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp

from AsyncGPT4All import AsyncGPT4All
from callback import StreamingLLMCallbackHandler
from constants import CHROMA_SETTINGS
from fastapi import WebSocket
import os

load_dotenv()

llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')


class PrivateGPT:
    def __init__(self):
        llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
        db = Chroma(persist_directory=persist_directory, embedding_function=llama, client_settings=CHROMA_SETTINGS)
        self.retriever = db.as_retriever()
        self.qa = None

    def set_llm(self, websocket: WebSocket):
        # Prepare the LLM
        callbacks = [StreamingLLMCallbackHandler(websocket)]
        if model_type == "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        elif model_type == "GPT4All":
            llm = AsyncGPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        else:
            raise ValueError(f"Model {model_type} not supported!")
        self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=self.retriever,
                                              return_source_documents=True)

    def complete(self, completion_input: str):
        res = self.qa(completion_input)
        answer, docs = res['result'], res['source_documents']
        return answer, docs

