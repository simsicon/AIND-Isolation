"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import pdb
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return custom_score_1(game, player)

def custom_score_1(game, player, weight=1.2):
    """Calculate the score by rewarding the gap between my moves and opponent moves
    with exponential.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    weight : float
        A float number to weight how much opponent's moves should be valued.

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = len(game.get_legal_moves(player))
    opponent = game.get_opponent(player)
    opponent_moves = len(game.get_legal_moves(opponent))

    return math.exp(my_moves - weight * opponent_moves)

def custom_score_2(game, player, weight=1.2):
    """Calculate the score by rewarding the gap between my moves and opponent moves
    with polynomial.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = len(game.get_legal_moves(player))
    opponent = game.get_opponent(player)
    opponent_moves = len(game.get_legal_moves(opponent))
    x = my_moves - weight * opponent_moves
    return x * x

def custom_score_3(game, player):
    """Calculate the score by the density of blank spaces.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    local_blanks_count = 0
    _last_move = game.__last_player_move__[player]

    for i in range(_last_move[0] - 2, _last_move[0] + 2, 1):
        if i > 0:
            for j in range(_last_move[1] - 2, _last_move[1] + 2, 1):
                if j > 0:
                    if game.__board_state__[i][j] == game.BLANK:
                        local_blanks_count += 1

    global_blanks_count = len(game.get_blank_spaces())

    return local_blanks_count / global_blanks_count


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if len(legal_moves) == 0:
            return (-1, -1)

        # Occupy the center as possible
        # if ((game.height == game.width)
        #     and (game.height % 2 == 1)
        #     and (game.width % 2 == 1)):
        #
        #     center_cell = (game.height // 2 + 1, game.width // 2 + 1)
        #     if (center_cell in legal_moves and
        #         game.__board_state__[center_cell[0]][center_cell[1]] == 0):
        #         return center_cell

        def rotate_90(origin_state):
            _h, _w = len(origin_state), len(origin_state[0])

            state_90 = []
            for i in range(_h):
                _row = []
                for j in range(_w):
                    _row.append(origin_state[_h - j - 1][i])
                state_90.append(_row)
            return state_90

        def rotate(origin_game):
            origin_state = origin_game.__board_state__
            state_90 = rotate_90(origin_state)
            state_180 = rotate_90(state_90)
            state_270 = rotate_90(state_180)
            return origin_state, state_90, state_180, state_270

        def search_method_helper(search_depth):
            if self.method == 'minimax':
                _score, _move = self.minimax(game, search_depth)
            elif self.method == 'alphabeta':
                _score, _move = self.alphabeta(game, search_depth)
            return _score, _move

        _best_move = None
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                _depth = 1
                while True:
                    _, _best_move = search_method_helper(_depth)
                    _depth += 1
            else:
                _, _best_move = search_method_helper(self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return _best_move
        # Return the best move from the last completed search iteration
        return _best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        _legal_moves = game.get_legal_moves()
        if len(_legal_moves) == 0 or depth == 0:
            return (self.score(game, self), None)

        _scores = []
        for move in _legal_moves:
            next_state = game.forecast_move(move)
            child_score, _ = self.minimax(next_state, depth - 1, not maximizing_player)
            _scores.append((child_score, move))

        if maximizing_player:
            return max(_scores, key= lambda x : x[0])
        else:
            return min(_scores, key= lambda x : x[0])

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        _legal_moves = game.get_legal_moves()
        if len(_legal_moves) == 0 or depth == 0:
            return  (self.score(game, self), None)

        _scores = []
        for move in _legal_moves:
            next_state = game.forecast_move(move)
            child_score, _ = self.alphabeta(next_state, depth - 1,
                                            alpha, beta, not maximizing_player)

            _scores.append((child_score, move))

            # prune
            if ((maximizing_player and child_score >= beta) or
               (not maximizing_player and child_score <= alpha)):
                break

            # update current alpha and beta
            if maximizing_player and child_score > alpha:
                alpha = child_score
            elif not maximizing_player and child_score < beta:
                beta = child_score

        if maximizing_player:
            return max(_scores, key= lambda x : x[0])
        else:
            return min(_scores, key= lambda x : x[0])
