import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from utils import save_checkpoint, load_checkpoint, print_examples
from transformers import AutoTokenizer
from model import CNNtoRNN
from dataset import FlickrDataset
import evaluate  



def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = FlickrDataset('flickr', 'captions.txt', tokenizer, transform= transform)
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    torch.backends.cudnn.benchmark = True
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = tokenizer.vocab_size
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 10

    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)
            outputs = model(imgs, captions)
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        
        print(f"Epoch {epoch}\{num_epochs}, loss {loss.item()}")
        # Validation loop
        model.eval()
        val_loss = 0
        references = []
        hypotheses = []
        with torch.no_grad():
            for imgs, captions in tqdm(val_loader, total=len(val_loader), leave=False):
                imgs = imgs.to(device)
                captions = captions.to(device)
                outputs = model(imgs, captions)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
                )
                val_loss += loss.item()

                predicted_captions = outputs.argmax(dim=2)  # Get the most likely token
                for i in range(len(captions)):
                    ref = tokenizer.decode(
                        captions[i].tolist(), skip_special_tokens=True
                    )
                    hyp = tokenizer.decode(
                        predicted_captions[i].tolist(), skip_special_tokens=True
                    )
                    
                    references.append([ref])  
                    hypotheses.append(hyp)  

        # Calculate BLEU scores
        bleu = evaluate.load("bleu")
        bleu_scores = bleu.compute(predictions=hypotheses, references=references)
        print(
            f"Epoch {epoch+1}/{num_epochs}, BLEU-1: {bleu_scores['precisions'][0]:.4f}, BLEU-2: {bleu_scores['precisions'][1]:.4f}, BLEU-3: {bleu_scores['precisions'][2]:.4f}, BLEU-4: {bleu_scores['precisions'][3]:.4f}"
        )
        print(
            f"Epoch {epoch+1}/{num_epochs}, BLEU-1: {bleu_scores['precisions'][0]:.4f}, BLEU-2: {bleu_scores['precisions'][1]:.4f}, BLEU-3: {bleu_scores['precisions'][2]:.4f}, BLEU-4: {bleu_scores['precisions'][3]:.4f}"
        )

        val_loss /= len(val_loader)
        writer.add_scalar("Validation loss", val_loss, global_step=epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
        writer.add_scalar("BLEU-1", bleu_scores["precisions"][0], global_step=epoch)
        writer.add_scalar("BLEU-2", bleu_scores["precisions"][1], global_step=epoch)
        writer.add_scalar("BLEU-3", bleu_scores["precisions"][2], global_step=epoch)
        writer.add_scalar("BLEU-4", bleu_scores["precisions"][3], global_step=epoch)


if __name__ == "__main__":
    train()
