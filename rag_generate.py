import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from transformers import BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline


class LLMGenerator:

    def __init__(
        self,
        llm_model_name="meta-llama/Llama-3-8B-Instruct",
        device="cuda",
        max_new_tokens=300,
    ):
        self.llm_model_name = llm_model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._llm = None

    def _get_llm(self) -> HuggingFacePipeline:
        if self._llm is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

            if self.llm_model_name == "meta-llama/Llama-3-8B-Instruct":
                model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    quantization_config=bnb_config,
                    device_map="auto" if self.device != "cpu" else None,
                )
            
            elif self.llm_model_name == "google/flan-t5-small":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.llm_model_name,
                    quantization_config=bnb_config,
                    device_map="auto" if self.device != "cpu" else None,
                )

            hf_pipe = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,   # deterministic for evaluation
                temperature=0.0,
                return_full_text=False,
            )

            self._llm = HuggingFacePipeline(pipeline=hf_pipe)

        return self._llm
    
    def generate_answer(
        self,
        docs,
        prompt,
        question,
    ):
        """
        Generate an answer for a single question using RAG.
        """

        llm = self._get_llm()
        response = llm.invoke(prompt)

        return {
            "question": question,
            "generated_answer": response,
            "retrieved_papers": list(
                {d.metadata.get("paper_id") for d in docs}
            ),
            "retrieved_chunks": docs,
        }
