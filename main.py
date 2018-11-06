import mcts
import nimsim
import numpy as np

# MCTS is always player 0, starting player is defined in nimsim (0 or 1)

player_id = 0
ns = nimsim.NimSim(20, 5, starting_player=player_id)
mcts = mcts.MCTS(ns, 100, player_id)

n_games = 100
wins = 0
for i in range(n_games):
    print('game', i)
    ns.reset(player_id)
    done = False
    state = ns.state
    while not done:
        if state[0] == player_id:
            action = mcts.pick_action(ns.state)
            state, done = ns.step(action)
        else: # opponent move
            state, done = ns.step(np.random.choice(ns.action_space))
        
    if state[0] != player_id:
        wins += 1

print('wins: {}/{}'.format(wins, n_games))