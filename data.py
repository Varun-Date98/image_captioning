import os
import spacy
import torch
import pandas as pd
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    """
    Vocabulary class for converting words to numbers and getting strings from prediction vectors
    """
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build(self, sentences):
        idx = 4
        counts = {}

        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                counts[word] = counts.get(word, 0) + 1

                if counts[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokens = self.tokenizer_eng(text)
        return [
            self.stoi[token]
            if token in self.stoi else self.stoi["<UNK>"]
            for token in tokens
        ]


class FlickrData(Dataset):
    """
    Custom PyTorch dataset class to load images and their captions from Flickr 8k dataset.
    """
    def __init__(self, root_dir, captions, transforms=None, frequency_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions)
        self.transforms = transforms

        # Getting the images and captions
        self.images = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize and build vocabulary
        self.vocab = Vocabulary(frequency_threshold)
        self.vocab.build(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_path = self.images[index]
        img = Image.open(os.path.join(self.root_dir, img_path)).convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        # Create numerical vector of the captions
        caption_vector = [self.vocab.stoi["<SOS>"]]
        caption_vector += self.vocab.numericalize(caption)
        caption_vector.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(caption_vector)


class Normalize:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_index)
        return imgs, targets


def get_loader(root_dir,
               captions_file,
               transform,
               batch_size=32,
               num_workers=2,
               shuffle=True,
               pin_memory=True):
    dataset = FlickrData(root_dir, captions_file, transforms=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=Normalize(pad_index=pad_idx)
    )

    return loader, dataset
