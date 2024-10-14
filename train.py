import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from data import get_loader
from model import CNNtoRNN
from utils import save_checkpoint, load_checkpoint, print_examples


def train():
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataloader, dataset = get_loader(
        root_dir="./flickr8k/Images",
        captions_file="./flickr8k/captions.txt",
        transform=transform,
        num_workers=2
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = True
    save_model = True

    # Hyper parameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    lr = 1e-3
    num_epochs = 50

    # tensor board
    writer = SummaryWriter("runs/flicker")
    step = 0

    # Model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }
            save_checkpoint(checkpoint)

        for idx, (image, captions) in enumerate(train_dataloader):
            image = image.to(device)
            captions = captions.to(device)

            outputs = model(image, captions[:-1])
            loss = loss_fn(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Train Loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()
