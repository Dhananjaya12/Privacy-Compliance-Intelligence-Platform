from __future__ import annotations

import json
import os
from typing import List, Optional

import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


class ChunkingManager:

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        config: dict = {},
        device: str = "cpu",
    ):
        self.embedding_model_name = embedding_model
        self.llm_model_name       = llm_model
        self.device               = device
        self.config               = config
        self.api_key              = os.getenv("HUGGING_FACE_API")

        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._llm: Optional[tuple]                         = None

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )
        return self._embeddings

    def _get_llm(self) -> tuple:
        if self._llm is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name, token=self.api_key
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                token=self.api_key,
                quantization_config=bnb_config,
                device_map="auto" if self.device != "cpu" else None,
            )

            llm_pipeline = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                do_sample=True,
                temperature=self.config["llm"]["temperature"],
                repetition_penalty=1.1,
                max_new_tokens=self.config["llm"]["max_new_tokens"],
                return_full_text=False,
                truncation=True,
            )

            self._llm = (llm_pipeline, tokenizer)

        return self._llm

    # ── Chunking strategies ───────────────────────────────────────────────────

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
                chunks      = splitter.split_documents([doc])
                page_lines  = []
                page_chunks = []

                for sec_idx, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=chunk.page_content,
                        metadata={
                            **doc.metadata,
                            "chunk_strategy": "token",
                            "section_id":     sec_idx,
                            "doc_index":      doc_idx,
                        },
                    )
                    page_chunks.append(chunk_doc)
                    page_lines.append(
                        json.dumps(
                            {"text": chunk_doc.page_content, "metadata": chunk_doc.metadata},
                            ensure_ascii=False,
                        )
                    )

                if page_lines:
                    f.write("\n".join(page_lines) + "\n")
                    f.flush()
                    all_chunks.extend(page_chunks)

        return all_chunks

    def semantic_chunking(
        self,
        documents: List[Document],
        output_path,
        append: bool = True,
    ) -> List[Document]:
        embeddings = self._get_embeddings()
        splitter   = SemanticChunker(embeddings=embeddings)

        all_chunks = []
        mode = "a" if append else "w"

        with open(output_path, mode, encoding="utf-8") as f:
            for doc_idx, doc in enumerate(tqdm(documents, desc="Semantic chunking")):
                chunks      = splitter.split_documents([doc])
                page_lines  = []
                page_chunks = []

                for sec_idx, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=chunk.page_content,
                        metadata={
                            **doc.metadata,
                            "chunk_strategy": "semantic",
                            "section_id":     sec_idx,
                            "doc_index":      doc_idx,
                        },
                    )
                    page_chunks.append(chunk_doc)
                    page_lines.append(
                        json.dumps(
                            {"text": chunk_doc.page_content, "metadata": chunk_doc.metadata},
                            ensure_ascii=False,
                        )
                    )

                if page_lines:
                    f.write("\n".join(page_lines) + "\n")
                    f.flush()
                    all_chunks.extend(page_chunks)

        return all_chunks

    def agentic_chunking(
        self,
        documents: List[Document],
        output_path,
        append: bool = True,
    ) -> List[Document]:
        all_chunks = []
        mode = "a" if append else "w"

        with open(output_path, mode, encoding="utf-8") as f:
            for doc_idx, doc in enumerate(tqdm(documents, desc="Agentic chunking")):
                llm_pipeline, tokenizer = self._get_llm()
                sections    = self._llm_split_sections(doc.page_content, llm_pipeline, tokenizer)
                page_lines  = []
                page_chunks = []

                for sec_idx, section in enumerate(sections):
                    chunk = Document(
                        page_content=section,
                        metadata={
                            **doc.metadata,
                            "chunk_strategy": "agentic",
                            "section_id":     sec_idx,
                            "doc_index":      doc_idx,
                        },
                    )
                    page_chunks.append(chunk)
                    page_lines.append(
                        json.dumps(
                            {"text": chunk.page_content, "metadata": chunk.metadata},
                            ensure_ascii=False,
                        )
                    )

                if page_lines:
                    f.write("\n".join(page_lines) + "\n")
                    f.flush()
                    all_chunks.extend(page_chunks)

        return all_chunks

    # ── Agentic helpers ───────────────────────────────────────────────────────

    def _build_agentic_prompt(self, text: str) -> list:
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert research assistant. "
                    "Your task is to segment academic text into coherent topical sections. "
                    "Each section should group related paragraphs that discuss the same idea."
                ),
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
                ),
            },
        ]

    def _llm_split_sections(
        self, text: str, llm_pipeline, tokenizer
    ) -> List[str]:
        prompt = self._build_agentic_prompt(text)

        formatted_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = llm_pipeline(formatted_prompt)[0]["generated_text"]

        return [
            s.strip()
            for s in output.split("###")
            if len(s.strip()) > 100
        ]
