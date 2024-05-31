from model import StockwishEvalMLP


class Node:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state  # should be bitmap vector
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.num_visits = 0

    def value(self, value_network):
        # value_network should be the trained MLP model which can give a value between (-1, 1) for any given position
        # TODO: do we need to normalize value_network to (-1, 1)?
        return value_network(self.state)


def run_mcts(initial_state, num_iterations=10):
    # create new leaf node
    # root = Node(initial_state)
    # node = root
    # if node.is_leaf_node():
    #     if node.num_visits == 0:
    #         result = rollout()

    pass



