from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


class LLMGenerator:
    """
    Thin wrapper around a quantised HuggingFace causal LM.
    Used for Phase 1 RAG generation and answer grading.

    All compliance LLM calls (Phase 2) go through Azure OpenAI directly
    in compliance_nodes.py and are not routed here.
    """

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cpu",
        config: dict = {},
    ):
        self.llm_model_name = llm_model_name
        self.device         = device
        self.config         = config
        self.api_key        = os.getenv("HUGGING_FACE_API")
        self._llm: HuggingFacePipeline | None = None

    def _get_llm(self) -> HuggingFacePipeline:
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

            hf_pipe = pipeline(
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

            self._llm = HuggingFacePipeline(pipeline=hf_pipe)

        return self._llm

    def generate_answer(
        self,
        docs: List[Any],
        prompt: str,
        question: str,
        mode: str = "answer",
    ) -> Dict[str, Any]:
        """
        Generate a response from the local LLM.

        mode="answer" — standard generation using invoke()
        mode="grade"  — short numeric output (max 2 tokens, greedy)
        """
        llm = self._get_llm()

        if mode == "grade":
            response = llm.pipeline(
                prompt,
                max_new_tokens=2,
                do_sample=False,
                temperature=0.0,
                top_p=1,
                return_full_text=False,
            )[0]["generated_text"].strip()
        else:
            response = llm.invoke(prompt)

        return {
            "question":        question,
            "generated_answer": response,
            "retrieved_papers": list(
                {d.metadata.get("paper_id") for d in docs if hasattr(d, "metadata")}
            ),
            "retrieved_chunks": docs,
        }
