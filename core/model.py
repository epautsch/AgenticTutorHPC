from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)


console = Console()


def load_hf_model_and_processor(
    model_id: str,
    *,
    device_map: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a text-generation chat model and its processor/tokeinzer in a backend-agnostic way.
    Works with:
        - Gemma3
        - OSS

    Returns:
        (hf_model, processor_or_tokenizer)
    """
    console.print(f"[bold green]Loading model:[/bold green] {model_id}")

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code
        ).eval()
    except Exception as e:
        if "gemma-3" in model_id:
            from transformers import Gemma3ForConditionalGeneration

            hf_model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation="eager",
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            ).eval()
        else:
            raise e

    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if not hasattr(processor, "apply_chat_template"):
            raise OSError("Processor lacks chat template; falling back to tokenizer.")
    except Exception:
        processor = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

    try:
        if getattr(processor, "pad_token_id", None) is None and getattr(processor, "eos_token_id", None) is not None:
            processor.pad_token = processor.eos_token
    except Exception:
        pass

    return hf_model, processor


def _as_string_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@dataclass
class LLMClient:
    hf_model: Any
    processor: Any
    system_prompt: str
    max_new_tokens: int

    def _build_inputs(self, user_prompt: str) -> Any:
        """
        Create model-ready inputs from (system, user) messages.
        """
        rich_messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
        ]

        simple_messages = _as_string_messages(self.system_prompt, user_prompt)

        def _to_model_inputs(msgs):
            return self.processor.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        raw = None
        if hasattr(self.processor, "apply_chat_template"):
            try:
                raw = _to_model_inputs(rich_messages)
            except Exception:
                raw = _to_model_inputs(simple_messages)
        else:
            text = f"{self.system_prompt}\n\n{user_prompt}\n"
            raw = self.processor(
                text,
                return_tensors="pt",
            )

        inputs = {}
        for k, v in raw.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.hf_model.device)
            else:
                inputs[k] = v
        return inputs

    def generate(self, user_prompt: str) -> str:
        #input = [
        #    {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
        #    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        #]
        # DEBUG print
        #console.print("[blue]▶️  LLMClient.generate() input:[/]\n", input)

        with console.status("Generating response...", spinner="dots"):
            # tokenize & run
            #raw = self.processor.apply_chat_template(
            #    input,
            #    add_generation_prompt=True,
            #    tokenize=True,
            #    return_dict=True,
            #    return_tensors="pt"
            #).to(self.hf_model.device, dtype=torch.bfloat16)
            # )

            # new for 1b model
            #for k, v in raw.items():
            #    if isinstance(v, torch.Tensor):
            #        raw[k] = v.to(self.model.device, dtype=torch.bfloat16)
            inputs = self._build_inputs(user_prompt)

            # DEBUG print
            console.print("[blue]▶️  LLMClient.generate() tokenized input:[/]\n", inputs)

            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    # cache_implementation="offloaded",
                    do_sample=False
                )
            # decode
            gen_ids = out[0][input_len:]
            decoded = self.processor.decode(gen_ids, skip_special_tokens=True)
            console.print("[red]▶️  LLMClient.generate() output:[/]\n", decoded)
            return decoded
