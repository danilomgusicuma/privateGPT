from fastapi import FastAPI, Body

from _typing.completion import CompletionInput, CompletionOutput
from privateGPT import PrivateGPT

app = FastAPI()
private_gpt = PrivateGPT()


@app.get("/")
async def root():
    return "Running privateGPT api!"


@app.post("/completion", response_model=CompletionOutput)
async def completion(completion_input: CompletionInput = Body(...)):
    answer, docs = private_gpt.complete(completion_input.message)
    return CompletionOutput(answer=answer, docs=docs)
