import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from src.core.config import val_dataset_len


class StreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_len=1024, stride=1024, batch_size=4):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.batch_size = batch_size
        self.buffer = []
        
    def __iter__(self):
        batch_x, batch_y = [], []
        
        for example in self.dataset:
            text = example.get('text', '')
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            self.buffer.extend(token_ids)
            
            while len(self.buffer) >= self.max_len:
                chunk = self.buffer[:self.max_len]
                self.buffer = self.buffer[self.stride:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                batch_x.append(x)
                batch_y.append(y)
                
                if len(batch_x) == self.batch_size:
                    yield torch.stack(batch_x), torch.stack(batch_y)
                    batch_x, batch_y = [], []
                    
def get_data_loader(dataset, split, streaming, tokenizer_name="gpt2"):
    raw_dataset = load_dataset(dataset, split=split, streaming=streaming)
    
    train_stream = raw_dataset.skip(val_dataset_len) # type: ignore
    val_stream = raw_dataset.take(val_dataset_len) # type: ignore
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_dataset = StreamingDataset(train_stream, tokenizer)
    val_dataset = StreamingDataset(val_stream, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)
    return train_loader, val_loader