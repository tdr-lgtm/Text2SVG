import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from . import config

def load_tokenizer():
    # Load the pretrained tokenizer and ensure a padding token is set.
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model():
    # Load the pretrained Qwen causal language model, optionally using
    # 4-bit NF4 quantization to reduce memory usage.
    quant = None
    if config.LOAD_IN_4BIT:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, # quantise the scales too
        )

    return AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

def resize_embeddings(model, bridge):
    """
    Resize the model's embedding table to accommodate the SVG vocabulary.

    Parameters:
        model: The pretrained language model.
        bridge: The vocabulary bridge containing the required SVG token IDs.

    Returns:
        The total number of rows in the resized embedding table.
    """
    model.resize_token_embeddings(bridge.hi, pad_to_multiple_of=64)

    total = model.get_input_embeddings().weight.shape[0]
    model.config.vocab_size = total

    return total

def freeze_base_embeddings(model, base_vocab):
    """
    Freeze the pretrained embedding rows while allowing newly added embedding rows to be trained.

    Gradient hooks zero the gradients of the original vocabulary rows
    during backpropagation. If the input and output embeddings are tied,
    only one hook is needed.

    Parameters:
        model: The language model whose embeddings will be configured.
        base_vocab: The number of original vocabulary rows to freeze.

    Returns:
        True if the input and output embeddings share the same weights,
        otherwise False.
    """
    embed_in = model.get_input_embeddings()
    embed_out = model.get_output_embeddings()
    tied = embed_out.weight.data_ptr() == embed_in.weight.data_ptr()

    def zero_base_rows(grad):
        grad = grad.clone()
        grad[:base_vocab] = 0
        return grad

    embed_in.weight.requires_grad = True
    embed_in.weight.register_hook(zero_base_rows)

    if not tied:
        embed_out.weight.requires_grad = True
        embed_out.weight.register_hook(zero_base_rows)

    return tied

def attach_lora(model):
    """
    Prepare the model for quantized training and attach LoRA adapters.

    Parameters:
        model: The pretrained language model.

    Returns:
        The model wrapped with LoRA adapters.
    """
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False # required with checkpointing

    return get_peft_model(model, LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.LORA_TARGETS),
    ))

def build_optimizer(model):
    """
    Build an AdamW optimizer with separate parameter groups for LoRA adapters and embeddings.

    Embeddings use weight_decay=0 because AdamW applies weight decay
    independently of the gradient. Therefore, zeroing gradients alone
    does not protect parameters from weight decay.

    The function prefers bitsandbytes AdamW8bit and falls back to
    PyTorch AdamW if bitsandbytes is unavailable.

    Parameters:
        model: The language model with its LoRA adapters attached.

    Returns:
        A tuple containing:
            optimizer: The configured optimizer.
            lora_count: Number of LoRA parameter tensors.
            embedding_count: Number of embedding parameter tensors.
            kind: Name of the optimizer implementation used.
    """
    embed_in = model.get_input_embeddings()
    embed_out = model.get_output_embeddings()

    embedding_ids = {id(embed_in.weight)}
    if embed_out.weight.data_ptr() != embed_in.weight.data_ptr():
        embedding_ids.add(id(embed_out.weight))

    embedding_params, lora_params = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        target = (embedding_params if id(parameter) in embedding_ids else lora_params)
        target.append(parameter)

    groups = [
        {"params": lora_params,
         "weight_decay": config.WEIGHT_DECAY, 
         "lr": config.LR_LORA},
        
        {"params": embedding_params,
         "weight_decay": 0.0, 
         "lr": config.LR_EMBEDDING},
    ]

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.999), eps=1e-8)
        kind = "AdamW8bit"
    except ImportError:
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        kind = "AdamW"

    return optimizer, len(lora_params), len(embedding_params), kind

def build(bridge):
    """
    The function loads the pretrained model, resizes its embedding table
    for the SVG vocabulary, attaches LoRA adapters, configures the
    embedding gradients, and creates the optimizer.

    Parameters:
        bridge: The vocabulary bridge containing the SVG token mappings
            and the size of the original pretrained vocabulary.

    Returns:
        A tuple containing:
            model: The configured language model.
            optimizer: The optimizer used for training.
            info: A dictionary containing model and optimizer details.
    """
    model = load_model()

    total_vocab = resize_embeddings(model, bridge)
    model = attach_lora(model)
    tied = freeze_base_embeddings(model, bridge.base_vocab)

    optimizer, n_lora, n_embed, kind = build_optimizer(model)

    return model, optimizer, {
        "total_vocab": total_vocab,
        "tied": tied,
        "lora_tensors": n_lora,
        "embedding_tensors": n_embed,
        "optimizer": kind,
    }