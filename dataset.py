import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, tokenizer, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            captions_file (string): File path for the captions text file.
            tokenizer (transformers tokenizer): Tokenizer to encode the captions.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_files = []
        self.captions = []
        
        # Load the captions and image file paths
        with open(os.path.join(root_dir, captions_file), 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(',')
                image_file = parts[0].split()[0]
                caption = ','.join(parts[1:]).strip()
                image_path = os.path.join(root_dir, 'images', image_file)
                if os.path.exists(image_path):
                    self.image_files.append(image_path)
                    self.captions.append(caption)
                
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        caption = self.captions[idx]
        encoded_caption = self.tokenizer(caption, padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids.squeeze(0)
        
        return image, encoded_caption

# Usage
