import numpy as np


class BatchSampler:
    def __init__(self, batch_size, data, bins, length=None):
        self.data_lens = list(sorted([(dat.shape[0], i) for i, dat in enumerate(data)]))
        
        self.bin_ids = np.array_split(np.arange(len(self.data_lens)), bins)
        self.batch_size = batch_size 
        self.bins = bins
        self.length = length
        
    def __len__(self):
        if self.length is None:
            return (len(self.data_lens) + self.batch_size - 1) // self.batch_size
        
        return self.length
    
    def __iter__(self):
        for i in range(len(self)):
            bin_id = np.random.randint(self.bins)

            rand_ids = np.random.choice(self.bin_ids[bin_id], self.batch_size, replace=False)

            orig_ids = [self.data_lens[rand_id][1] for rand_id in rand_ids]

            yield orig_ids
