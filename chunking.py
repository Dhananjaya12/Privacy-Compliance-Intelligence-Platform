from typing import List, Optional
from langchain_core.documents import Document

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TokenTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import HuggingFacePipeline

import json
from tqdm import tqdm

class ChunkingManager:
    """
    Efficient chunking manager with lazy-loaded components.
    Heavy models are only loaded if the corresponding strategy is used.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "google/flan-t5-small",#"meta-llama/Llama-3.1-8B-Instruct", #"meta-llama/Llama-2-7b-chat-hf", 
        device: str = "cpu"
    ):
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.device = device

        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._llm: Optional[HuggingFacePipeline] = None

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )
        return self._embeddings

    def semantic_chunking(
    self,
    documents: List[Document],
    output_path,
    append: bool = True,
) -> List[Document]:

        embeddings = self._get_embeddings()
        splitter = SemanticChunker(embeddings=embeddings)

        all_chunks = []
        mode = "a" if append else "w"

        with open(output_path, mode, encoding="utf-8") as f:
            for doc_idx, doc in enumerate(tqdm(documents, desc="Semantic chunking")):

                chunks = splitter.split_documents([doc])
                page_lines = []
                page_chunks = []

                for sec_idx, chunk in enumerate(chunks):

                    chunk_doc = Document(
                        page_content=chunk.page_content,
                        metadata={
                            **doc.metadata,
                            "chunk_strategy": "semantic",
                            "section_id": sec_idx,
                            "doc_index": doc_idx,
                        }
                    )

                    page_chunks.append(chunk_doc)

                    page_lines.append(json.dumps({
                    "text": chunk_doc.page_content,
                    "metadata": chunk_doc.metadata,
                }, ensure_ascii=False))

                if page_lines:
                    f.write("\n".join(page_lines) + "\n")
                    f.flush()
                    all_chunks.extend(page_chunks)
                

        return all_chunks

    def token_chunking(
    self,
    documents: List[Document],
    output_path,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    append: bool = True,
) -> List[Document]:

        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks = []
        mode = "a" if append else "w"

        with open(output_path, mode, encoding="utf-8") as f:
            for doc_idx, doc in enumerate(tqdm(documents, desc="Token chunking")):

                chunks = splitter.split_documents([doc])
                page_lines = []
                page_chunks = []

                for sec_idx, chunk in enumerate(chunks):
                    chunk_doc = Document(
                    page_content=chunk.page_content,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "token",
                        "section_id": sec_idx,
                        "doc_index": doc_idx,
                    }
                )
                    # chunk.metadata["chunk_strategy"] = "token"
                    page_chunks.append(chunk_doc)

                    page_lines.append(json.dumps({
                        "text": chunk_doc.page_content,
                        "metadata": chunk_doc.metadata,
                    }, ensure_ascii=False))

                if page_lines:
                    f.write("\n".join(page_lines) + "\n")
                    f.flush()
                    all_chunks.extend(page_chunks)
                
        return all_chunks
    
    def _build_agentic_prompt(self, text: str):
            return [
                {
                    "role": "system",
                    "content": (
                        "You are an expert research assistant. "
                        "Your task is to segment academic text into coherent topical sections. "
                        "Each section should group related paragraphs that discuss the same idea."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Segment the following academic text into coherent sections.\n\n"
                        "Rules:\n"
                        "- Do NOT summarize or paraphrase.\n"
                        "- Do NOT copy the prompt text.\n"
                        "- Preserve original wording.\n"
                        "- Each section should be 1–3 paragraphs.\n"
                        "- Separate sections using ONLY the delimiter ###.\n\n"
                        f"TEXT:\n{text}"
                    )
                }
            ]

        
    def agentic_chunking(self, documents: List[Document], output_path, prompt_template, append: bool = True,) -> List[Document]:
        all_chunks = []
        mode = "a" if append else "w"

        with open(output_path, mode, encoding="utf-8") as f:
            for doc_idx, doc in enumerate(tqdm(documents, desc="Agentic chunking")):
                # print('doc_idx, doc', doc_idx, doc)
                # sections = self._llm_split_sections(doc.page_content, llm)
                prompt = prompt_template.format(text=doc.page_content)
                llm_pipeline, tokenizer = self._get_llm()
                sections = self._llm_split_sections(doc.page_content, llm_pipeline, tokenizer, prompt)
                # print('sections', sections)
                page_lines = [] 
                page_chunks = []
                
                for sec_idx, section in enumerate(sections):
                    chunk = Document(
                        page_content=section,
                        metadata={
                            **doc.metadata,
                            "chunk_strategy": "agentic",
                            "section_id": sec_idx,
                            "doc_index": doc_idx,
                        }
                    )

                    page_chunks.append(chunk)

                    page_lines.append(json.dumps({
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                    }, ensure_ascii=False))

                    # prepare JSONL line (unchanged schema)
                if page_lines:
                    f.write("\n".join(page_lines) + "\n")
                    f.flush()
                    all_chunks.extend(page_chunks)
        return all_chunks
    
    def _llm_split_sections(self, text: str, llm_pipeline, tokenizer, prompt):
    
        formatted_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
    
        output = llm_pipeline(formatted_prompt)[0]["generated_text"]
    
        sections = [
            s.strip()
            for s in output.split("###")
            if len(s.strip()) > 100
        ]
    
        return sections