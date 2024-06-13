from fastapi import FastAPI
import chromadb
from chromadb.config import Settings
import uvicorn

app = FastAPI()

@app.on_event("startup")
def startup_event():
    global chroma_client
    # Configura aquí la conexión a Chroma DB según sea necesario
    chroma_client = chromadb.HttpClient(host="localhost", port=8000, settings=Settings(allow_reset=True))

@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run("chroma_server:app", host="0.0.0.0", port=8000, reload=True)


# Añade aquí más rutas para manejar operaciones específicas con Chroma DB

