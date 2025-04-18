from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

class RookieVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, client,config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client,config=config)