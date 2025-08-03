import yaml
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.decoder import DecoderBlock
from src.data.loader import get_data_loaders
from src.utils.generate import generate_text
from src.utils.loss import calc_loss_batch, calc_loss_loader


torch.manual_seed(123)


def train(model, train_loader, val_loader, n_epochs, optimizer, tokenizer, max_new_tokens, contenxt, eval_freq=100, save_freq=1000, device='cpu'):
    model.train()
    train_losses, val_losseses = [], []
    tokens_seen = 0
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        for idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            X = X.to(device=device)
            y = y.to(device=device)
            loss = calc_loss_batch(X, y, model, device=device)
            loss.backward()
            optimizer.step()
            
            tokens_seen += X.numel()
            if idx%eval_freq == 0:
                val_loss = calc_loss_loader(val_loader, model, device=device)
                train_losses.append(loss.item())
                val_losseses.append(val_loss)
                print(f"Step {idx}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Tokens seen: {tokens_seen:,}")
            
            if idx%save_freq == 0:
                torch.save(model.state_dict(), f"{model.model_name}/step_{idx}.pt")
                generate_text(model, tokenizer, max_new_tokens, contenxt, device=device)
    return train_losses, val_losseses, tokens_seen


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Train a transformer model from scratch using native Pytorch.")
    parser.add_argument('--config', type=str, default='gpt2.yaml', required=True, help='Path to model configuration file.')
    args = parser.parse_args()
    
    # config_file = r"D:\PythonProjects\llm-training\mini_morph\models\gpt2.yaml"
    config_file = args.config
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    sys.exit(0)
    if config['dtype'] == 'bfloat16':
        config['dtype'] = torch.bfloat16
    else:
        config['dtype'] = torch.float16
    
    model = DecoderBlock(config)
    model.to(device=config['device'])
    print(f"Model summary:\n{model}\n")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    # Account for weight tying
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")
    
    # # with torch.no_grad():
    # X = torch.tensor([[1, 2, 3]])
    # X = X.to(device=config['device'])
    
    # # X = X.unsqueeze(0)
    # print(model(X))
    # print(X.shape)

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    
    train_loader, val_loader = get_data_loaders(
        dataset=config['dataset'],
        split=config['split'],
        streaming=config['streaming'],
        tokenizer=tokenizer
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    train_losses, val_losseses, tokens_seen = train(model, train_loader, val_loader, n_epochs=config['n_epochs'], optimizer=optimizer, tokenizer=tokenizer, max_new_tokens=config['max_new_tokens'], contenxt=config['context'], eval_freq=config['eval_freq'], save_freq=config['save_freq'], device=config['device'])