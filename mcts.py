from model import StockwishEvalMLP
import numpy as np
import chess
import random

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
        
        self.ucb = 0 
        self.N = 0 #Num of times parent node has been visited
        self.n = 0 #Num of times current node has been visited
        self.v = 0 #Exploitation factor: higher v = higher success rate

    def value(self, value_network):
        # value_network should be the trained MLP model which can give a value between (-1, 1) for any given position
        # TODO: do we need to normalize value_network to (-1, 1)?
        return value_network(self.state)

def ucb(curr_node, c=2):
  return curr_node.v + c*np.sqrt(np.log(curr_node.N + 1**(-10))/(curr_node.n + 1**(-10))) #c is the exploration constant, higher = more random moves 

def selection(curr_node):
    # Select the child node with the highest UCB value
    best_child = curr_node.children[0]
    for child in curr_node.children:
        if child.ucb > best_child.ucb:
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
    curr_node.n += 1
    curr_node.v += reward
    while(curr_node.parent != None):
        curr_node.N += 1
        curr_node = curr_node.parent
    return curr_node

def run_mcts(initial_state = chess.STARTING_FEN, num_iterations=10):
    # create new leaf node
    root = Node(initial_state)
    node = root
    expansion(node, True)
    for child in node.children:
        print(child)
        
    # if node.is_leaf_node():
    #     if node.num_visits == 0:
    #         result = rollout()

    pass

run_mcts()


