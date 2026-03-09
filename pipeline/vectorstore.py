from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

try:
    from langchain_postgres import PGVector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False


class VectorStoreFactory:
    def __init__(
        self,
        vs_type: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        pgvector_connection: Optional[str] = None,
    ):
        self.vs_type = vs_type.lower()
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.pgvector_connection = pgvector_connection

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        self.vectorstore = None

    def build_or_load(self, documents: List[Document]):
        if self.vs_type == "faiss":
            self._build_faiss(documents)

        elif self.vs_type == "chroma":
            self._build_chroma(documents)

        elif self.vs_type == "pgvector":
            self._build_pgvector(documents)

        else:
            raise ValueError(f"Unknown vector store type: {self.vs_type}")

        return self.vectorstore

    def as_retriever(self, k: int = 5):
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def _build_faiss(self, documents: List[Document]):
        if self.persist_dir and self.persist_dir.exists():
            print("✅ Loading existing FAISS index")
            self.vectorstore = FAISS.load_local(
                str(self.persist_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("🔹 Building FAISS index")
            self.vectorstore = FAISS.from_documents(
                documents,
                embedding=self.embeddings
            )
            if self.persist_dir:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self.vectorstore.save_local(str(self.persist_dir))

    def _build_chroma(self, documents: List[Document]):
        if not self.persist_dir or not self.collection_name:
            raise ValueError("Chroma requires persist_dir and collection_name")

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        print("🔹 Initializing Chroma")
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
        )

        if self.vectorstore._collection.count() == 0:
            print("🔹 Adding documents to Chroma")

            batch_size = 5400   # Must be < 5461
            total = len(documents)

            for i in range(0, total, batch_size):
                batch = documents[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                print(f"✅ Added {min(i + batch_size, total)}/{total}")

            print("✅ Finished building Chroma collection")
        else:
            print("✅ Existing Chroma collection found — skipping rebuild")

    def _build_pgvector(self, documents: List[Document]):
        if not PGVECTOR_AVAILABLE:
            raise RuntimeError("PGVector is not installed")

        if not self.pgvector_connection or not self.collection_name:
            raise ValueError("PGVector requires connection string and collection_name")

        print("🔹 Initializing PGVector")
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.pgvector_connection,
            use_jsonb=True,
        )

        self.vectorstore.add_documents(documents)
