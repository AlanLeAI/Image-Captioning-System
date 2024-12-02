from model import *
from get_loader import *


transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

dataset  = FlickrDataset("archive/images","archive/captions.txt")
vocab_size = len(dataset.vocab)
model = CNNtoRNN(32, 128, vocab_size, 1)
state_dict = torch.load("my_checkpoint.pth")
model.load_state_dict(state_dict["state_dict"])


model.eval()
test_img1 = transform(Image.open("test_examples/dog.jpeg").convert("RGB")).unsqueeze(0)
print("Example 1 CORRECT: Dog on a beach by the ocean")
print(
    "Example 1 OUTPUT: "
    + " ".join(model.caption_image(test_img1, dataset.vocab))
)

