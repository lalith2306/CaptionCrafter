import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import tqdm
import csv

# Dataset path
IMAGE_DIR = r"C:\Users\Lenovo\Downloads\ImgCap\Images"
CAPTIONS_FILE = r"C:\Users\Lenovo\Downloads\ImgCap\captions.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleTokenizer:
    def __init__(self, captions, max_caption_length=20):
        self.word2idx = {}
        self.idx2word = {}
        self.max_caption_length = max_caption_length
        self.build_vocab(captions)

    def build_vocab(self, captions):
        all_words = set()
        for caption in captions:
            all_words.update(caption.lower().split())

        for idx, word in enumerate(sorted(all_words), 1):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.word2idx['<pad>'] = 0
        self.idx2word[0] = '<pad>'

    def caption_to_indices(self, caption):
        indices = [self.word2idx[word] for word in caption.lower().split() if word in self.word2idx]
        return indices[:self.max_caption_length]

    def pad_caption(self, indices):
        return indices + [0] * (self.max_caption_length - len(indices))

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.captions = self.load_captions(captions_file)
        self.image_names = list(self.captions.keys())
        self.tokenizer = tokenizer

    def load_captions(self, captions_file):
        captions = {}
        with open(captions_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if len(row) != 2:
                    print(f"Skipping improperly formatted line: {row}")
                    continue
                img_name, caption = row[0].strip(), row[1].strip()
                captions[img_name] = caption
        print(f"Loaded {len(captions)} captions.")
        return captions

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        caption = self.captions[img_name]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption_indices = self.tokenizer.caption_to_indices(caption)
        caption_indices = self.tokenizer.pad_caption(caption_indices)

        return image, torch.tensor(caption_indices, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

captions = []
with open(CAPTIONS_FILE, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  
    for row in reader:
        if len(row) == 2:
            captions.append(row[1].strip())


tokenizer = SimpleTokenizer(captions)
dataset = ImageCaptionDataset(IMAGE_DIR, CAPTIONS_FILE, tokenizer, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#resnet to extract feature
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 256)
        
    def forward(self, images):
        features = self.feature_extractor(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# lstm for caption generation
class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CaptionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, pad_token_idx=0):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, captions in tqdm.tqdm(dataloader):
            images = images.to(device)
            captions = captions.to(device)
            outputs = model(images, captions[:, :-1]) 
            batch_size, seq_len_minus1, vocab_size = outputs.size()

            # Flatten outputs and targets
            outputs = outputs.view(-1, vocab_size) 
            targets = captions[:, 1:].reshape(-1) 
            mask = targets != pad_token_idx
            print(f"Epoch {epoch+1}, Batch Loss Calculation:")
            print(f"  outputs shape: {outputs.shape}, targets shape: {targets.shape}, mask shape: {mask.shape}")

            valid_indices = mask.nonzero(as_tuple=True)[0]  
            filtered_outputs = outputs[valid_indices]
            filtered_targets = targets[mask]
            print(f"  filtered_outputs shape: {filtered_outputs.shape}, filtered_targets shape: {filtered_targets.shape}")


            if filtered_outputs.size(0) == 0 or filtered_targets.size(0) == 0:
                print("Warning: No valid targets in this batch!")
                continue

            # loss ,backwardpass,optimization
            loss = criterion(filtered_outputs, filtered_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader)}')

    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict()
    }, 'image_captioning_model.pth')


vocab_size = len(tokenizer.word2idx)
encoder = ImageEncoder().to(device)
decoder = CaptionDecoder(vocab_size=vocab_size, embed_size=256, hidden_size=512, num_layers=1).to(device)
model = ImageCaptioningModel(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_model(model, dataloader, criterion, optimizer, num_epochs=10)
