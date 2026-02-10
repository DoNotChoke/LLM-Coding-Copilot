from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

import torch

from typing import List, Optional

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

specials_to_add = []
for tok in (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE):
    if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
        specials_to_add.append(tok)
    

if specials_to_add:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": specials_to_add},
        replace_additional_special_tokens=False,
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    attn_implementation="sdpa",
    trust_remote_code=True
)

if specials_to_add:
    model.resize_token_embeddings(len(tokenizer))

model = model.to(device)
model.eval()

DEFAULT_STOP_STRINGS: List[str] = [
    "```",               # markdown fence
    "\n\n\n",            # too many blank lines
    "<|endoftext|>",     # end marker (some tokenizers)
]

def encode_stop_strings(stop_strings: List[str]) -> List[List[int]]:
    encoded: List[List[int]] = []
    for s in stop_strings:
        if not s:
            continue
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            encoded.append(ids)
    return encoded


class StopOnSequences(StoppingCriteria):
    """
    Stop generation when the tail of input_ids matches any stop-sequence token ids.
    Works for multi-token stop strings.
    """
    def __init__(self, stop_sequences_token_ids: List[List[int]]):
        super().__init__()
        self.stop_seqs = [seq for seq in stop_sequences_token_ids if seq]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids is None or input_ids.numel() == 0:
            return False
        seq = input_ids[0].tolist()
        for stop_ids in self.stop_seqs:
            n = len(stop_ids)
            if n <= len(seq) and seq[-n:] == stop_ids:
                return True
        return False

def build_fim_prompt(prefix: str, suffix: str) -> str:
    prefix = prefix or ""
    suffix = suffix or ""
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def strip_at_stop_strings(text: str, stop_strings: List[str]) -> str:
    cut = None
    for s in stop_strings:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)

    if cut is not None:
        return text[:cut]
    return text


async def generate(
        prefix: str,
        suffix: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: bool = True,
        stop: Optional[List[str]] = None,
):
    prompt = build_fim_prompt(prefix, suffix)
    stop_strings = DEFAULT_STOP_STRINGS + stop

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    stop_token_id_seqs = encode_stop_strings(stop_strings)
    stopping = StoppingCriteriaList([StopOnSequences(stop_token_id_seqs)]) if stop_token_id_seqs else None

    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
            use_cache=True
        )

    gen_ids = gen[0]
    new_ids = gen_ids[input_ids.shape[1]:]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)

    completion = strip_at_stop_strings(completion, stop_strings)

    completion = completion.rstrip("\n\r\t ")

    finish_reason = "stop"
    if len(new_ids) >= max_new_tokens:
        finish_reason = "length"

    return completion, finish_reason