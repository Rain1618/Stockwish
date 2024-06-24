import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


# after output from our prediction, we want to convert back to cp
# using max and min from original dataset
def convert_to_cp(pred, min, max):
    return (pred*max)+min



def plot_loss_over_epochs(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    epoch_num = checkpoint['epoch']
    losses = checkpoint['loss_values']
    val_losses = checkpoint['valid_losses']
    print("Model successfully loaded!")
    assert epoch_num == len(losses)
    losses_n = [loss.item() for loss in losses]
    val_n = [val_losses.item() for v_loss in val_losses]
    fig, ax = plt.subplots()
    ax.plot(range(len(losses_n)), losses_n)
    ax.plot(range(len(val_n)), val_n)
    ax.legend(['Training', 'Validation'])
    ax.set_title('Losses over epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.set_xlabel('Epoch')
    plt.show()


def calculate_validation_loss_epoch(model, device, val_loader):
    model.eval()
    loop = tqdm(val_loader)
    losses = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.squeeze(1).float().to(device), target.float().unsqueeze(1).to(device)
            predictions = model(data)
            loss = nn.MSELoss()(predictions, target)
            losses.append(loss)
    return sum(losses) / len(losses) if losses else 0


def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1, 8, 8)

def fen_to_vector(fen_string):
    """
    Converts a FEN string to a 768-bit bitboard vector representation.

    Args:
        fen_string: The FEN string representing the chess position.

    Returns:
        A 12 x 8 x 8 numpy array representing the bitboard vector.
    """
    board = chess.Board(fen_string)

    black, white = board.occupied_co

    bitboards = np.array([
        black & board.pawns,
        black & board.knights,
        black & board.bishops,
        black & board.rooks,
        black & board.queens,
        black & board.kings,
        white & board.pawns,
        white & board.knights,
        white & board.bishops,
        white & board.rooks,
        white & board.queens,
        white & board.kings,
        ], dtype=np.uint64)

    # we need to encode the information for which turn it is
    board_array = bitboards_to_array(bitboards).astype('int8')
    if board.turn == chess.WHITE:
        # change all of the arrays of black pieces to -1
        for i in range(6):
            board_array[i] *= -1
    else:
        for i in range(6, 12):
            # change all of the arrays of white pieces to -1
            board_array[i] *= -1

    return array_to_vec(board_array)

def array_to_vec(board_array):
    arr = board_array.reshape(1, 768)
    return arr

if __name__ == '__main__':
    print(fen_to_vector(chess.STARTING_FEN))



