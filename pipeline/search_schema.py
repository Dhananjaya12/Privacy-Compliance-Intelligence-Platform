"""
pipeline/search_schema.py

Single source of truth for the Azure AI Search index field schema used by
BOTH the regulations index and the policies index.

Why an explicit schema?
-----------------------
LangChain's default `AzureSearch` index stores all metadata as one opaque JSON
string in a `metadata` field, which is NOT filterable per-key. To scope retrieval
to a single document (`paper_id eq '...'`) or document class (`doc_type eq 'policy'`)
we must declare those metadata keys as first-class **filterable** fields.

When the same key appears in a Document's `metadata` dict at upload time,
LangChain copies it into the matching top-level field, so an OData `$filter`
on `paper_id` / `regulation` / `doc_type` / `page` / `chunk_id` works.

The first four fields (id, content, content_vector, metadata) mirror LangChain's
defaults exactly so the rest of the AzureSearch integration keeps working.
"""

from __future__ import annotations

from typing import List

# all-MiniLM-L6-v2 produces 384-dimensional embeddings.
DEFAULT_EMBEDDING_DIM = 384

# LangChain AzureSearch default field names (do not rename — the integration
# references these constants internally).
FIELD_ID = "id"
FIELD_CONTENT = "content"
FIELD_CONTENT_VECTOR = "content_vector"
FIELD_METADATA = "metadata"

# Vector-search profile name LangChain creates by default.
VECTOR_PROFILE_NAME = "myHnswProfile"

# Metadata keys promoted to filterable top-level fields.
FILTERABLE_METADATA_FIELDS = ("paper_id", "regulation", "doc_type", "page", "chunk_id")


def build_compliance_fields(embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> List:
    """
    Return the explicit field list to pass as `fields=` to `AzureSearch(...)`.

    Imported lazily so that modules importing this file (e.g. in notebooks or
    test collection) don't require the azure-search SDK unless they actually
    build an index.
    """
    from azure.search.documents.indexes.models import (
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SimpleField,
    )

    return [
        # ── LangChain defaults ────────────────────────────────────────────────
        SimpleField(
            name=FIELD_ID,
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name=FIELD_CONTENT,
            type=SearchFieldDataType.String,
        ),
        SearchField(
            name=FIELD_CONTENT_VECTOR,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
        SearchableField(
            name=FIELD_METADATA,
            type=SearchFieldDataType.String,
        ),
        # ── Filterable metadata (the whole point of this schema) ───────────────
        SimpleField(
            name="paper_id",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="regulation",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="doc_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="page",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),
        SimpleField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
    ]
