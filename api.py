import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA

from _typing.completion import ChatResponse
from ingest import DocumentLoader
from privateGPT import PrivateGPT

app = FastAPI()
private_gpt = PrivateGPT()
loader = DocumentLoader()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return "Running privateGPT api!"


@app.websocket("/completion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    private_gpt.set_llm(websocket)
    while True:
        try:
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            retriever: RetrievalQA = private_gpt.qa
            response = await retriever.acall({"query": question})

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())

        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        filename = file.filename
        contents = file.file.read()
        loader.load_doc_content(
            doc_name=filename,
            content=contents
        )
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}
