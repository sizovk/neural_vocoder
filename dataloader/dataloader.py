from base import BaseDataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch):
    batch = pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch


class LJSpeechDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, validation_split=0.0, num_workers=0):
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_batch)
