from torch import nn


class BatchCollator:
    def __call__(self, samples):
        data = nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=0)
        
        return data.long()