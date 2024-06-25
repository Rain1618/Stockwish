import chess
from mcts import run_mcts, load_model, Node
from model import StockwishEvalMLP

MODEL_PATH = "models/model_chess_4.pth"
# TODO: MODIFY THE C constant (exploration vs. exploitation constant), right now I think the C is too high, try with lower values
# TODO: The selection process might be broken. Imagine c=0, then we want to pick the child node with the MOST NEGATIVE VALUE
# TODO: since that means that this is the worse state for the opponent.
# TODO: Shouldn't UCB be v/n + c*sqrt(log(...))? right now it is just v + c*sqrt((...)) https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
def play():
    board = chess.Board()
    model = load_model(MODEL_PATH, type=StockwishEvalMLP)
    node = Node(chess.STARTING_FEN)
    print("Welcome to Stockwish!")
    while not board.is_game_over():

        print(board)
        if board.turn == chess.WHITE:
            #move = input("Enter move: ")
            node = run_mcts(model, node, num_iterations=1000)
            move = str(node.parent_action)
            board.push_san(move)
        else:
            node = run_mcts(model, node, num_iterations=1000)
            board.push_san(str(node.parent_action))
    print(board)
    print("Game over")

play()