import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

from pipeline.search_schema import build_compliance_fields


class VectorStoreFactory:

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_name: str = "pdf-rag-index",
        top_k: int = 5,
        use_compliance_schema: bool = True,
    ):
        self.index_name = index_name
        self.top_k = top_k
        # When True, the index is created with explicit filterable fields
        # (paper_id, regulation, doc_type, page, chunk_id) so the compliance
        # retriever can scope searches with an OData $filter.
        self.use_compliance_schema = use_compliance_schema

        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_KEY")

        if not endpoint or not key:
            raise ValueError(
                "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY must be set in .env"
            )

        self.endpoint = endpoint
        self.key = key

        print("🔹 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self.vectorstore: Optional[AzureSearch] = None

    def build_or_load(self, documents: List[Document], force: bool = False):
        """
        Connect to (or create) the Azure AI Search index and add documents.

        force=False (default): skip upload if index already has documents.
                               Used on server startup — just connect.
        force=True           : always upload regardless of existing count.
                               Used by kg_builder.py and ingest endpoints.
        """
        print(f"🔹 Connecting to Azure AI Search index '{self.index_name}'...")

        fields = build_compliance_fields() if self.use_compliance_schema else None

        self.vectorstore = AzureSearch(
            azure_search_endpoint=self.endpoint,
            azure_search_key=self.key,
            index_name=self.index_name,
            embedding_function=self.embeddings.embed_query,
            fields=fields,
        )

        # Check existing document count
        try:
            existing_count = self.vectorstore.client.get_document_count()
        except Exception:
            existing_count = 0

        print("Force", force)
        if not force and existing_count > 0:
            print(f"✅ Azure Search index already has {existing_count} documents — skipping upload.")
        elif not documents:
            print(f"✅ No documents to upload — index has {existing_count} documents.")
        else:
            print(f"🔹 Uploading {len(documents)} documents to Azure AI Search...")
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                print(f"  ✅ Uploaded {min(i + batch_size, len(documents))}/{len(documents)}")
            print(f"✅ All {len(documents)} documents uploaded.")

        return self.vectorstore

    def as_retriever(self, k: Optional[int] = None):
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_or_load() first.")
        return self.vectorstore.as_retriever(
            search_type="hybrid",
            search_kwargs={"k": k or self.top_k},
        )