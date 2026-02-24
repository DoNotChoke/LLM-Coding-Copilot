from transformers import AutoTokenizer
from datasets import load_dataset

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from tqdm.auto import tqdm
from uuid import uuid4

from dotenv import load_dotenv

from embedding import Embedder

load_dotenv()

class CodeIngestion:
    def __init__(
            self,
            collection,
            code = None,
            embedder: Embedder = None,
            tokenizer: AutoTokenizer = None,
            text_splitter = None,
            batch_limit: int = 100
    ):
        self.collection = collection
        self.code = code