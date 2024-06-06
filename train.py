import torch
import chess
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from importlib import reload


# local imports
from model import StockwishEvalMLP
from dataset import ChessDataset, Split
from utils import calculate_validation_loss_epoch, convert_to_cp



# Hyper parameters
LEARNING_RATE = 1e-2 # modified from 1e-3
MOMENTUM = 0.9 # modified from 0.7
NESTEROV = True
BATCH_SIZE = 1024 # modified from 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 2
NUM_EPOCHS = 80
PIN_MEMORY = True
LOAD_MODEL = True
ROOT_DIR = "data"
MODEL_PATH = "drive/MyDrive/model_chess.pth"

# Model specific params
NUM_FEATURES = 8 * 8 * 12  # bitmap representation of chessboard is 8 x 8 x num_pieces = 8x8x12
NUM_HIDDEN = 2048  # from paper

# Globals
epoch_num = 0
losses = []
valid_losses = []

def visualize_results(model: StockwishEvalMLP, root_path, num_results):
    # need to set model to eval mode
    model.eval()
    # loading in val dataset with fens so that we can visualize results
    transform = lambda x: (torch.tensor(x)).float()
    val_ds = ChessDataset(root_path=root_path, transform=transform, split=Split.VALID, return_fen=True)
    print(val_ds.min)
    print(val_ds.max)

    for i in range(num_results):
        data, target, fen = val_ds[i]
        data = data.unsqueeze(0).to(device=DEVICE)
        #target = target.float().unsqueeze(1).to(device=DEVICE)
        print(data.size())
        pred = model(data).item()
        loss = (pred - target) ** 2
        cp_pred = convert_to_cp(pred, val_ds.min, val_ds.max)
        cp_target = convert_to_cp(target, val_ds.min, val_ds.max)
        board = chess.Board(fen)
        #display(board) for colab
        print(board)
        print(f"MSE Loss: {loss}, Predicted CP: {cp_pred}, Actual CP: {cp_target}")
    model.train()



def get_loaders(
        root_dir,
        batch_size,
        train_transform,
        target_transform,
        num_workers=4,
        pin_memory=True,
):

    train_ds = ChessDataset(root_path=root_dir, transform=train_transform, target_transform=target_transform, split=Split.TRAIN)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = ChessDataset(root_path=root_dir, transform=train_transform, target_transform=target_transform, split=Split.VALID)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


"""
Does one epoch of training.
"""

def train(train_loader, val_loader, model, optimizer, loss_fn, scaler):
    global epoch_num
    global losses
    global valid_losses
    model.train()

    loop = tqdm(train_loader)

    for batch_idx, (data, targets) in enumerate(loop):

        batch_losses = []
        data = data.squeeze(1).to(device=DEVICE) #[b_size, 768]
        #print(data)
        #print(data.size())
        targets = targets.float().unsqueeze(1).to(device=DEVICE) #[b_size, 1]
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            batch_losses.append(loss)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

    # once epoch is finished, we save state
    epoch_num += 1
    avg_epoch_loss =sum(batch_losses)/len(batch_losses)
    losses.append(avg_epoch_loss)
    # we should keep track of our validation losses as well
    loss = calculate_validation_loss_epoch(model, DEVICE, val_loader)
    valid_losses.append(loss)
    print(f"Epoch {epoch_num} completed. Avg. Training loss: {avg_epoch_loss} Avg. validation loss: {loss}. Model successfully saved!")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'epoch': epoch_num,
        'loss_values': losses,
        'valid_losses': valid_losses,
    }, MODEL_PATH)

def main():
    global epoch_num
    global losses
    global valid_losses

    train_transform = lambda x: (torch.tensor(x)).float()
    target_transform = None
    train_loader, val_loader = get_loaders(ROOT_DIR, BATCH_SIZE, train_transform, target_transform, NUM_WORKERS,
                                           PIN_MEMORY)
    loss_fn = nn.MSELoss()
    model = StockwishEvalMLP(num_features=NUM_FEATURES, num_units_hidden=NUM_HIDDEN, num_classes=1).to(DEVICE)
    optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=NESTEROV)
    if LOAD_MODEL:
        # Loading a previous stored model from MODEL_PATH variable
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch_num = checkpoint['epoch']
        losses = checkpoint['loss_values']
        valid_losses = checkpoint['valid_losses']
        print("Model successfully loaded!")

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS - epoch_num):
        train(train_loader, val_loader, model, optimizer, loss_fn, scaler)

if __name__ == '__main__':
    main()