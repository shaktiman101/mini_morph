import torch


def generate_tokens(model, token_ids, max_new_tokens, context_size):
    # tokens is (B, T) array of indices in current context
    for _ in range(max_new_tokens):
        # trim context if it exceeds context size
        trimmed_token_ids = token_ids[:, -context_size:]
        
        with torch.no_grad():
            logits = model(trimmed_token_ids)
            
        # logits is (B, T, vocab_size)
        logits = logits[:, -1, :]
        token_idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)
        token_ids = torch.cat((token_ids, token_idx_next), dim=1)   # (B, T+1)
        
    return token_ids


def generate_text(model, tokenizer, max_new_tokens, context, device):
    model.eval()
    
    # revisit
    context_size = model.pos_emb.weight.shape[0]
    token_ids = tokenizer.batch_encode_plus(context, return_tensors='pt', padding=True).device(device)
    new_token_ids = generate_tokens(model, token_ids['input_ids'], max_new_tokens=max_new_tokens, context_size=context_size)
    
    generated_text = tokenizer.batch_decode(new_token_ids.tolist())
    print(f"Generated text: {generated_text}")
    model.train()


def generate_text2(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(-torch.inf).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if eos_id is not None and idx_next.item() == eos_id:
            break  # Stop generating early if end-of-sequence token is encountered and eos_id is specified

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx