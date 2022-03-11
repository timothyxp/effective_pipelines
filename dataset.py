import torch


class TokenDataset:
    def __init__(self, data, pad_max_len=None):
        self.data = data
        self.pad_max_len = pad_max_len or -1
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.pad_max_len -1 and item.shape[0] < self.pad_max_len:
            item = torch.cat((
                item,
                torch.zeros(self.pad_max_len - item.shape[0])
            ))
        elif item.shape[0] > self.pad_max_len:
            item = item[:self.pad_max_len]
            
        return item.long()
