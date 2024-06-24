import chess
from mcts import run_mcts, load_model
from model import StockwishEvalMLP

MODEL_PATH = "models/model_chess_4.pth"

def play():
    board = chess.Board()
    model = load_model(MODEL_PATH, type=StockwishEvalMLP)
    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            move = input("Enter move: ")
            board.push_san(move)
        else:
            node = run_mcts(model, initial_state=board.fen(), num_iterations=1)
            board.push_san(str(node.parent_action))
    print(board)
    print("Game over")

play()