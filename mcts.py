import nimsim
import numpy as np
import math

class MCTS:
    # This is an implementation of MCTS that is focused on readability over performance

    class Node:
        def __init__(self, parent, action, env_output):
            self.action = action
            self.parent = parent
            self.state, self.done = env_output
            self.children = []
            self.visits = 1
            self.wins = 0

        def policy_value(self, opponent):
            qsa = self.wins / (1 + self.visits)
            usa = math.sqrt(math.log(self.parent.visits) / (1 + self.visits))
            return qsa + (usa if not opponent else -usa)

        quality = property(fget=lambda self : self.wins / (1 + self.visits))

    def __init__(self, env, M, player_id):
        self.env = env
        self.root = self.Node(None, None, (env.state, False))
        self.M = M
        self.player_id = player_id # to allow self-play, instead of hard-coding player 1

    def pick_action(self, state): # M full simulation runs
        self.root = self.Node(None, None, (self.env.state, False))
        
        for _ in range(self.M):
            self.tree_search(self.root)
        
        return self.root.children[np.argmax([child.quality for child in self.root.children])].action
        
    # return the index of the child to be chosen, according to the tree policy
    def choose_child(self, node):
        if node.state[0] == self.player_id:
            return np.argmax([child.policy_value(False) for child in node.children])
        return np.argmin([child.policy_value(True) for child in node.children])

    def tree_search(self, node):
        if len(node.children):
            self.tree_search(node.children[self.choose_child(node)])
        else:
            self.node_expansion(node)

    def node_expansion(self, node):
        node.children = [self.Node(node, action, self.env.simulate(action, node.state)) for action in self.env.action_space]
        for child in node.children:
            self.leaf_evaluation(child)

    def leaf_evaluation(self, node):
        done = False
        state = node.state
        while not done:
            state, done = self.env.simulate(np.random.choice(self.env.action_space), state)
        win = False
        if state[0] != self.player_id: # win (2-player)
            win = True
        self.backpropagation(node, win)

    def backpropagation(self, node, win):
        node.visits += 1
        if win:
            node.wins += 1
        if node.parent:
            self.backpropagation(node.parent, win)