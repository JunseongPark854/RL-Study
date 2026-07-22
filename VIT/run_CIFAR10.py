import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

from cnn import Classic_CNN
from vit_todo import ViT

MODEL_TYPE = "vit"

IMAGE_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 10
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://wandb.ai/pjs1540a-stony-brook-university/hw3
WANDB_ENTITY = "pjs1540a-stony-brook-university"
WANDB_PROJECT = "CIFAR10_CNN_ViT"
USE_WANDB = True 

# img -> input data
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), # PIL to float tensor
])

train_dataset = datasets.CIFAR10(
    root="./data/", train=True, transform=transform, download=True
)
test_dataset = datasets.CIFAR10(
    root="./data/", train=False, transform=transform, download=True
)
# root : 저장폴더, train = True/False에 따라 다른 data, transform = 위에서 정의한 변환순서
# download = True (없으면 다운)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
# 가져온 dataset을 배치로 묶어서 사용

def build_model(model_type):
    if model_type == "cnn":
        return Classic_CNN(
            image_size=IMAGE_SIZE,
            channels=CHANNELS,
            num_classes=NUM_CLASSES,
            base_channels=32,
            num_blocks=3,
            dropout=0.1,
        )
    if model_type == "vit":
        return ViT(
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            patch_size=(4,4), 
            num_classes=NUM_CLASSES,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            channels=CHANNELS,
            dim_head=16,
            dropout=0.1,
            emb_dropout=0.1,
        )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        y_pred = model(images.to(device))
        loss_out = criterion(y_pred, labels.to(device))
        loss_out.backward()
        optimizer.step()
        total_loss += loss_out.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device)
            y_pred = model(images.to(device)).argmax(dim=1)
            correct += (labels == y_pred).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    config = {
        "model_type": MODEL_TYPE,
        "image_size": IMAGE_SIZE,
        "channels": CHANNELS,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "device": str(DEVICE),
    }

    run = None
    if USE_WANDB:
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"mnist-{MODEL_TYPE}",
            config=config,
        )

    model = build_model(MODEL_TYPE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"device={DEVICE}, model={MODEL_TYPE}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        acc = evaluate(model, test_loader, DEVICE)
        print(f"epoch {epoch:02d} | loss={train_loss:.4f} | acc={acc*100:.2f}%")

        if run is not None:
            run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "test/accuracy": acc,
            })

    if run is not None:
        run.finish()