import yaml

import torch

from decoder import DecoderBlock

torch.manual_seed(123)
def train():
    pass


if __name__ == "__main__":
    config_file = r"D:\PythonProjects\llm-training\mini_morph\configs\models\custom.yaml"
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    
    if config['dtype'] == 'bfloat16':
        config['dtype'] = torch.bfloat16
    else:
        config['dtype'] = torch.float16
    
    gpt2_model = DecoderBlock(config)
    gpt2_model.to(device=config['device'])
    print(gpt2_model)
    
    # with torch.no_grad():
    X = torch.tensor([[1, 2, 3]])
    X = X.to(device=config['device'])
    
    # X = X.unsqueeze(0)
    print(gpt2_model(X))
    print(X.shape)

    total_params = sum(p.numel() for p in gpt2_model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Account for weight tying
    total_params_normalized = total_params - gpt2_model.tok_emb.weight.numel()
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")