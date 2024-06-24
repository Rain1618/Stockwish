import torch

from model import StockwishEvalMLP, StockwishEvalMLP_Mod
import numpy as np
import chess
import chess.svg
import random
from utils import fen_to_vector, convert_to_cp

#MODEL_PATH = "drive/MyDrive/model_chess_4.pth"  # model_4 is the best one so far
MODEL_PATH = "models/model_chess_4.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FEATURES = 8 * 8 * 12  # bitmap representation of chessboard is 8 x 8 x num_pieces = 8x8x12
NUM_HIDDEN = 2048  # changed from 2048
CP_MIN = -1000
CP_MAX = 2000


def load_model(path, type=StockwishEvalMLP):

    print(f"Loading model at: {path} ... ")
    if type == StockwishEvalMLP:
        model = StockwishEvalMLP(num_features=NUM_FEATURES, num_units_hidden=NUM_HIDDEN, num_classes=1).to(DEVICE)
    else:
        model = StockwishEvalMLP_Mod(num_features=NUM_FEATURES, num_units_hidden=1024, num_classes=1).to(DEVICE)
    checkpoint = torch.load(path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully!")
    return model


###
#https://jyopari.github.io/MCTS.html
#Each layer is player 1 or player 2's turn
###
class Node:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state  # FEN
        self.parent = parent #FEN
        self.parent_action = parent_action
        self.children = [] #List of Node of next possible actions

        #self.ucb = 0
        self.N = 0 #Num of times parent node has been visited
        self.n = 0 #Num of times current node has been visited
        self.v = 0 #Exploitation factor: higher v = higher success rate

    def value(self, value_network):
        # value_network should be the trained MLP model which can give a value between (-1, 1) for any given position
        # TODO: do we need to normalize value_network to (-1, 1)?
        return value_network(self.state)

    def is_leaf(self):
        return len(self.children) == 0

    def outcome(self):
        return chess.Board(self.state).outcome()

    def ucb(self, c=2):
        return self.v + c*np.sqrt(np.log(self.parent.n + 1**(-10))/(self.n + 1**(-10))) # c is the exploration constant, higher = more random moves

def selection(curr_node, c=2):
    # Select the child node with the highest UCB value
    best_child = curr_node.children[0]
    for child in curr_node.children:
        if child.ucb(c) > best_child.ucb(c):
            #print(child.ucb())
            #print(best_child.ucb())
            best_child = child
    return best_child

def expansion(curr_node):
    # Expand the current node by adding all possible child nodes
    #TODO: this gonna be a lot of possible future moves...
    #TODO: alternating white and black moves (double check)

    board = chess.Board(curr_node.state)
    next_moves = list(board.legal_moves)
    for move in next_moves: #Assuming no next moves if game over TODO: check assumption
        board.push(move)
        new_node = Node(board.fen(), curr_node, move)
        curr_node.children.append(new_node)
        board.pop()


def rollback(curr_node, reward): #reward = 1 if win, 0 if draw, -1 if loss
    #print(chess.Board(curr_node.state))
    #print(f"Current node reward {reward} ^")
    while(curr_node.parent != None):
        curr_node.n += 1
        curr_node.v = curr_node.v+reward if chess.Board(curr_node.state).turn == chess.WHITE else curr_node.v-reward
        curr_node.N += 1
        curr_node = curr_node.parent
    curr_node.n += 1
    curr_node.v += reward
    return curr_node


# this code is from https://github.com/niklasf/python-chess/discussions/864
def material_balance(board):
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    return (
            chess.popcount(white & board.pawns) - chess.popcount(black & board.pawns) +
            3 * (chess.popcount(white & board.knights) - chess.popcount(black & board.knights)) +
            3 * (chess.popcount(white & board.bishops) - chess.popcount(black & board.bishops)) +
            5 * (chess.popcount(white & board.rooks) - chess.popcount(black & board.rooks)) +
            9 * (chess.popcount(white & board.queens) - chess.popcount(black & board.queens))
    )

def decision(probability):
    return random.random() < probability

def get_eval(model, position):
    # position is a FEN string
    model.eval()
    position_vec = fen_to_vector(position)
    position_vec = (torch.tensor(position_vec)).float().to(DEVICE)
    pred = model(position_vec).item()
    return convert_to_cp(pred, CP_MIN, CP_MAX)


def find_next_best_legal_position(model, board, epsilon):
    is_whites_turn = board.turn
    legal_moves = board.legal_moves
    assert legal_moves.count() != 0
    best_move = None
    best_eval = CP_MIN if is_whites_turn else CP_MAX

    #print(legal_moves.count())
    for move in legal_moves:
        board.push(move)
        #display(board)
        position = board.fen()
        #print(board.fen())
        #print(get_eval(model, position))
        current_move_eval = get_eval(model, position)
        #print(current_move_eval)
        #print(move)
        if is_whites_turn:
            if current_move_eval > best_eval:
                best_move = move
                best_eval = current_move_eval
        else:
            if current_move_eval < best_eval:
                best_move = move
                best_eval = current_move_eval
        board.pop()
    # we should push the best_move found by model with probability (1-epsilon) + epsilon/(num_legal_moves)
    prob = (1-epsilon) + epsilon/(legal_moves.count())
    if decision(prob):
        board.push(best_move)
    else:
        move = random.choice(list(legal_moves))
        board.push(move)
    #print(best_eval)

def select_best_move(node, turn):
    list_of_v = [child.v for child in node.children]
    dict_of_v = {child.v: child for child in node.children}
    #return dict_of_v[min(list_of_v)] if turn else dict_of_v[max(list_of_v)]
    return dict_of_v[min(list_of_v)]


def simulate(initial_node, model, max_iterations=100, epsilon=1.0, display_board=False, seed=0):
    """
    initial_position: Node representing our initial position
    model: StockWishMLP class model which predicts value of any position
    max_iterations: how many iterations before we stop forcefully. One iteration = one half-move (e.g. a move for only one player)
    episilon: exploitation/exploration parameter, should be between 0 and 1, the higher the more we prioritize exploration
    display_board: whether we should display the board after each move
    seed: the seed for random() in order to get consistent results

    returns:
      - reward, terminal_node: either -1, 0, 1 and the terminal node

    TODO: new nodes should not be created
    """
    assert 1.0 >= epsilon >= 0
    random.seed(seed)
    current_node = initial_node
    current_position = initial_node.state
    board = chess.Board(current_position)

    iter_index = 0
    while (True):
        iter_index += 1
        find_next_best_legal_position(model, board, epsilon)
        #next_node = Node(next_position, current_node)
        #current_node.children.append(next_node)
        #current_node = next_node
        if display_board:
            #display(board)
            board
        if iter_index >= max_iterations or board.outcome():
            break
    #print(board.outcome())
    #display(board)
    if board.outcome() is None:
        # TODO: maybe change this to it returns a score reflecting the position eval
        #return 0
        eval = get_eval(model, board.fen())
        if eval > 500:
            return 1
        elif eval < -500:
            return -1
        else:
            return 0

    if board.outcome().winner is not None:
        return 1 if board.outcome().winner else -1
    else:
        # we want to avoid draws by repetition or 50 move rule
        if board.outcome().termination == chess.Termination.FIVEFOLD_REPETITION or board.outcome().termination == chess.Termination.FIFTY_MOVES:
            mat_balance = material_balance(board)
            if mat_balance >= 8:
                return 1
            elif mat_balance <= -8:
                return -1
            else:
                return 0
        else:
            return 0


def run_mcts(model, initial_state=chess.STARTING_FEN, num_iterations=10):
    # load model
    if model is None:
        model = load_model(MODEL_PATH, type=StockwishEvalMLP)
    # # create new leaf node
    # root = Node(initial_state)
    # node = root
    # expansion(node)
    # selected_node = selection(node)
    # result, node = simulate(selected_node.state, model)
    # rollback(node)
    # for child in node.children:
    #     pass

    # if node.is_leaf_node():
    #     if node.num_visits == 0:
    #         result = rollout()
    root = Node(initial_state)

    for i in range(num_iterations):
        # the selection function selects a node to run the simulation on
        # based on UCT and does expansions when needed
        node = root
        while not node.is_leaf():
            best_child = selection(node)
            node = best_child

        # check if node is a terminal node
        if node.outcome():
            if node.outcome().winner is not None:
                result = 1 if node.outcome().winner else -1
            else:
                result = 0
            #display(chess.Board(node.state))
            print(f"Iteration no. {i} done. Result of simulation: {result}")
            rollback(node, result)
            continue
        # if we reach here it means we have reached a leaf node, expand
        expansion(node)
        # select best child and simulate
        best_child = selection(node)
        node = best_child
        result = simulate(node, model)
        #rollback from selected node using the result
        #print(result)
        rollback(node, result)
        print(f"Iteration no. {i} done. Result of simulation: {result}")
    for child in root.children:
        #display(chess.Board(child.state))
        print(f"UCB: {child.ucb(c=0)}")
    return select_best_move(root, chess.Board(root.state).turn)

if __name__ == "__main__":
    fen = "rnbq1bnr/pppkpppN/7p/3p4/8/8/PPPPPPPP/RNBQKB1R w KQ - 2 4"
    # load model
    model = load_model(MODEL_PATH, type=StockwishEvalMLP)
    node = run_mcts(model, initial_state=fen, num_iterations=10)
    board = chess.Board(node.state)
    chess.svg.board(board)