import isolation
import game_agent

h, w = 7, 7  # board size
test_depth = 1
starting_location = (5, 3)
adversary_location = (0, 0)  # top left corner
iterative_search = False
search_method = "minimax"
heuristic = lambda g, p: 0.  # return 0 everywhere

# create a player agent & a game board
agentUT = game_agent.CustomPlayer(
    test_depth, game_agent.custom_score, iterative_search, search_method)
agentUT.time_left = lambda: 99  # ignore timeout for fixed-depth search
board = isolation.Board(agentUT, 'null_agent', w, h)

board.apply_move(starting_location)
board.apply_move(adversary_location)

print("current state")
print(board.to_string())
print("next states")
for move in board.get_legal_moves():
    next_state = board.forecast_move(move)
    print(next_state.to_string())
    agentUT.minimax(next_state, test_depth)
