from dotenv import load_dotenv
import uvicorn as uvicorn

if __name__ == "__main__":
    dotenv_path = './.env'
    load_dotenv(dotenv_path=dotenv_path)
    uvicorn.run("api:app", host="0.0.0.0", port=8088, reload=False)
