from contextlib import contextmanager
from pathlib import Path

from .core import digest, isolated_rng


GENERATION_POLICY = {
    "batch": "training: one problem; evaluation: fixed contiguous problem batches from config", "train_samples": 4,
    "evaluation_samples": 8, "greedy_separate": True,
    "training_rng": "continuous process RNG; checkpoint restores all RNG states",
    "evaluation_rng": "batch size 1: per-question seed; otherwise SHA256(evaluation seed, split, ordered batch question IDs); regenerate entire partial batch on resume; outer RNG restored",
    "determinism": "strict outside sampled top_p<1 generation; warn_only=True within that entire generate call, for ALL unsupported nondeterministic operations, not just cumsum; restore enabled and warn_only flags in finally",
    "padding": "retain first EOS; discard every subsequent token; EOS counts in loss",
}


@contextmanager
def generation_determinism(top_p, greedy):
    import torch
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if enabled and not greedy and top_p < 1:
            torch.use_deterministic_algorithms(True, warn_only=True)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)


def resolve_eos(tok, generation_config):
    """Use actual special-token mappings, never numeric model-specific constants."""
    vocab = tok.get_vocab()
    raw = generation_config.eos_token_id
    configured = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    ids = set(configured + [tok.eos_token_id])
    chat_end = "<|im_end|>"
    if chat_end in vocab:
        ids.add(vocab[chat_end])
    for token_id in ids:
        if not isinstance(token_id, int) or token_id not in tok.all_special_ids:
            raise ValueError(f"EOS {token_id!r} is not a verified tokenizer special token")
        token = tok.convert_ids_to_tokens(token_id)
        if vocab.get(token) != token_id or tok.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"EOS does not round-trip as one token: {token_id!r}/{token!r}")
    if tok.pad_token_id not in tok.all_special_ids:
        raise ValueError("Padding must be a verified special token")
    return {"policy": "union of generation_config EOS, tokenizer EOS and verified im_end if present",
            "generation_config_eos": raw, "tokenizer_eos": tok.eos_token_id,
            "eos_token_ids": sorted(ids), "tokens": {str(i): tok.convert_ids_to_tokens(i) for i in sorted(ids)},
            "pad_token_id": tok.pad_token_id, "vocabulary_hash": digest(vocab)}


def load_tokenizer(path, spec):
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    path = Path(path)
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    cfg = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    limits = [v for v in [getattr(cfg, "max_position_embeddings", None), tok.model_max_length,
                         spec.get("context_limit")] if isinstance(v, int) and 0 < v < 10**7]
    if not limits:
        raise ValueError("Cannot establish context limit; set model.context_limit from documented metadata")
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if (path / "generation_config.json").is_file():
        generation = GenerationConfig.from_pretrained(str(path), local_files_only=True)
        source = "generation_config.json"
    else:
        generation = GenerationConfig.from_model_config(cfg)
        source = "model config (generation_config.json absent)"
    tok.grpo_eos = {**resolve_eos(tok, generation), "source": source}
    return tok, min(limits)


def prompt_ids(tok, config, problem):
    text = config["prompt"]["template"].format(problem=problem)
    if config["prompt"]["mode"] == "chat":
        if not tok.chat_template:
            raise ValueError("Chat prompt requested but tokenizer has no chat template")
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
    if config["prompt"]["mode"] != "plain":
        raise ValueError("prompt.mode must be plain or chat")
    return tok.encode(text, add_special_tokens=True, truncation=False)


def validate_context(tok, limit, config, splits):
    out = {}
    for split, rows in splits.items():
        budget = config["train" if split == "train" else "evaluation"]["max_new_tokens"]
        lengths = [len(prompt_ids(tok, config, r["problem"])) for r in rows]
        for row, n in zip(rows, lengths):
            if n == 0 or n + budget > limit:
                raise ValueError(f"{split}/{row['id']}: input {n} + output {budget} exceeds context {limit}; no truncation")
        out[split] = {"min": min(lengths), "max": max(lengths), "output_budget": budget, "context": limit}
    return out


def adapter_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if p.requires_grad or "lora_" in name:
            h.update(name.encode())
            h.update(str(tuple(p.shape)).encode())
            h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def adapter_norms(model):
    import torch
    return {part + "_l2": float(torch.stack([p.detach().float().square().sum()
            for name, p in model.named_parameters() if p.requires_grad and part in name]).sum().sqrt())
            for part in ("lora_A", "lora_B")}


def load_model(path, config, lora_seed, device="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.pytorch_utils import Conv1D
    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["runtime"].get("attention_implementation", "eager"))
    supported = (torch.nn.Linear, Conv1D)
    explicit = config["lora"]["target_modules"]
    targets = explicit or [name for name, module in model.named_modules()
                           if isinstance(module, supported) and name.rsplit(".", 1)[-1] in config["lora"]["target_suffixes"]
                           and name != "lm_head"]
    modules = dict(model.named_modules())
    if not targets or any(name not in modules or not isinstance(modules[name], supported) for name in targets):
        raise ValueError("No supported attention/MLP linear targets, or invalid explicit target_modules; inspect architecture")
    if any("lm_head" in n or "embed" in n for n in targets):
        raise ValueError("Embedding/lm_head LoRA is outside this protocol")
    with isolated_rng(lora_seed):
        l = config["lora"]
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=l["r"], lora_alpha=l["alpha"],
                              lora_dropout=l["dropout"], target_modules=targets,
                              bias="none", init_lora_weights=True, use_rslora=False, use_dora=False))
    # Disable pretrained dropout too: generation and recomputed on-policy probabilities agree.
    dropout = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            if module.p:
                dropout.append({"name": name, "previous_p": module.p})
            module.p = 0.0
        for attr, value in list(vars(module).items()):
            if "dropout" in attr and isinstance(value, (int, float)) and value:
                dropout.append({"name": name + "." + attr, "previous_p": value})
                setattr(module, attr, 0.0)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n in trainable):
        raise ValueError("Only LoRA parameters may require gradients")
    model.to(device)
    if config["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.grpo_initial_adapter_hash = adapter_hash(model)
    return model, {"targets": targets, "trainable_parameters": trainable,
                   "attention_implementation": model.config._attn_implementation,
                   "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "trainable_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
                   "frozen_dtypes": sorted({str(p.dtype) for p in model.parameters() if not p.requires_grad}),
                   "disabled_dropout": dropout, "initial_adapter_hash": adapter_hash(model),
                   "initial_adapter_norms": adapter_norms(model)}


def generate(model, tok, ids, *, count, max_new_tokens, temperature, top_p, greedy=False):
    return generate_batch(model, tok, [ids], count=count, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_p=top_p, greedy=greedy)[0]


def generate_batch(model, tok, prompts, *, count, max_new_tokens, temperature, top_p, greedy=False):
    """Left-pad prompts; HF returns count consecutive responses per input prompt."""
    import torch
    from transformers import GenerationConfig
    was_training = model.training
    if not prompts or any(not ids for ids in prompts):
        raise ValueError("Generation requires nonempty tokenized prompts")
    results = [[] for _ in prompts]
    eos_meta = getattr(tok, "grpo_eos", None) or resolve_eos(tok, model.generation_config)
    eos_ids = set(eos_meta["eos_token_ids"])
    if greedy and count != 1:
        raise ValueError("Greedy evaluation is a separate single response")
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": not greedy,
              "num_beams": 1, "num_return_sequences": count,
              "repetition_penalty": 1.0, "eos_token_id": sorted(eos_ids),
              "pad_token_id": tok.pad_token_id, "bos_token_id": tok.bos_token_id,
              "use_cache": True}
    if not greedy:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=0)
    gc = GenerationConfig(**kwargs)
    device = next(model.parameters()).device
    width = max(map(len, prompts))
    x = torch.full((len(prompts), width), tok.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(x)
    for i, ids in enumerate(prompts):
        x[i, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[i, -len(ids):] = 1
    model.eval()
    try:
        with torch.inference_mode(), generation_determinism(top_p, greedy):
            sequences = model.generate(input_ids=x, attention_mask=mask, generation_config=gc)
            if len(sequences) != len(prompts) * count:
                raise ValueError("Generation returned an unexpected number of responses")
            for i, sequence in enumerate(sequences):
                tokens = sequence[width:].tolist()
                end = next((j for j, token in enumerate(tokens) if token in eos_ids), None)
                stopped = end is not None
                if stopped:
                    tokens = tokens[:end + 1]
                if not tokens:
                    raise ValueError("Empty token generation")
                if not stopped and len(tokens) != max_new_tokens:
                    raise ValueError("Generation ended without EOS before the configured length limit")
                results[i // count].append({"sample_index": i % count, "text": tok.decode(tokens, skip_special_tokens=True),
                                "token_ids": tokens, "generated_tokens": len(tokens),
                                "finish_reason": "eos" if stopped else "length",
                                "truncated": not stopped, "hit_token_budget": len(tokens) == max_new_tokens})
    finally:
        model.train(was_training)
    return results
