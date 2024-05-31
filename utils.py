import chess
import numpy as np

def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1, 8, 8)

def fen_to_array(fen_string):
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

    board_array = bitboards_to_array(bitboards)
    print(board_array.shape)
    return board_array

def array_to_vec(board_array):
    arr = board_array.reshape(1, 768)

    return arr