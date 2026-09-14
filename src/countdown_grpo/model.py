def load_model(config):
    import torch
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires an NVIDIA CUDA GPU with BF16 support")
    full_config = AutoConfig.from_pretrained(config.model, revision=config.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.model_revision, padding_side="left")
    if not tokenizer.chat_template:
        raise RuntimeError("The checkpoint must provide its official chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The Hub checkpoint has a vision tower; load only its language model.
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        config.model,
        revision=config.model_revision,
        config=full_config.text_config,
        key_mapping={r"^model\.language_model\.": "model."},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"Incomplete language-model weight loading: {info}")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.to("cuda")
    return model, tokenizer
