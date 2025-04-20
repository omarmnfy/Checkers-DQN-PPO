import numpy as np

class CheckersGame:
    """Custom Checkers game environment"""
    
    # Board piece representations
    EMPTY = 0
    RED = 1
    BLACK = 2
    RED_KING = 3
    BLACK_KING = 4

    def __init__(self):
        self.board = None
        self.current_player = None
        self.reset()

    def reset(self):
        """Initialize the game board"""
        self.board = np.zeros((8, 8), dtype=int)
        self.current_player = self.RED
        
        # Place black pieces (top of board)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row, col] = self.BLACK

        # Place red pieces (bottom of board)
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row, col] = self.RED

        return self.board.copy()

    def get_valid_moves(self, player):
        """Get all valid moves for the given player"""
        moves = []
        # First check for any available jumps (mandatory in checkers)
        for row in range(8):
            for col in range(8):
                if self.board[row, col] in [player, player + 2]:  # Regular or king piece
                    jumps = self._get_valid_jumps(row, col)
                    if jumps:
                        moves.extend(jumps)
        
        # If no jumps available, get regular moves
        if not moves:
            for row in range(8):
                for col in range(8):
                    if self.board[row, col] in [player, player + 2]:
                        moves.extend(self._get_valid_regular_moves(row, col))
        
        return moves

    def _get_valid_jumps(self, row, col):
        """Get all valid jump moves for a piece"""
        jumps = []
        piece = self.board[row, col]
        
        # Define possible jump directions based on piece type
        directions = []
        if piece == self.RED:
            directions.extend([(-2, -2), (-2, 2)])  # Up-left and up-right jumps only
        elif piece == self.BLACK:
            directions.extend([(2, -2), (2, 2)])    # Down-left and down-right jumps only
        elif piece in [self.RED_KING, self.BLACK_KING]:
            # Kings can jump in all four directions
            directions.extend([
                (-2, -2),  # Up-left
                (-2, 2),   # Up-right
                (2, -2),   # Down-left
                (2, 2)     # Down-right
            ])

        for dy, dx in directions:
            new_row, new_col = row + dy, col + dx
            mid_row, mid_col = row + dy//2, col + dx//2
            
            if self._is_valid_position(new_row, new_col):
                # Check if jump is valid
                if (self.board[new_row, new_col] == self.EMPTY and
                    self._is_opponent_piece(mid_row, mid_col, piece)):
                    jumps.append(((row, col), (new_row, new_col)))

        return jumps

    def _get_valid_regular_moves(self, row, col):
        """Get all valid regular moves for a piece"""
        moves = []
        piece = self.board[row, col]
        
        # Define possible move directions based on piece type
        directions = []
        if piece == self.RED:
            directions.extend([(-1, -1), (-1, 1)])  # Up-left and up-right only
        elif piece == self.BLACK:
            directions.extend([(1, -1), (1, 1)])    # Down-left and down-right only
        elif piece in [self.RED_KING, self.BLACK_KING]:
            # Kings can move in all four directions
            directions.extend([
                (-1, -1),  # Up-left
                (-1, 1),   # Up-right
                (1, -1),   # Down-left
                (1, 1)     # Down-right
            ])

        for dy, dx in directions:
            new_row, new_col = row + dy, col + dx
            if (self._is_valid_position(new_row, new_col) and 
                self.board[new_row, new_col] == self.EMPTY):
                moves.append(((row, col), (new_row, new_col)))

        return moves

    def _is_valid_position(self, row, col):
        """Check if position is within board boundaries"""
        return 0 <= row < 8 and 0 <= col < 8

    def _is_opponent_piece(self, row, col, piece):
        """Check if position contains an opponent's piece"""
        if not self._is_valid_position(row, col):
            return False
        target = self.board[row, col]
        return ((piece in [self.RED, self.RED_KING] and target in [self.BLACK, self.BLACK_KING]) or
                (piece in [self.BLACK, self.BLACK_KING] and target in [self.RED, self.RED_KING]))

    def make_move(self, move):
        """Execute a move on the board"""
        (start_row, start_col), (end_row, end_col) = move
        piece = self.board[start_row, start_col]
        
        # Move the piece
        self.board[start_row, start_col] = self.EMPTY
        self.board[end_row, end_col] = piece
        
        # Handle jumps (capture)
        if abs(end_row - start_row) == 2:
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            self.board[mid_row, mid_col] = self.EMPTY
        
        # Handle king promotion
        if piece == self.RED and end_row == 0:
            self.board[end_row, end_col] = self.RED_KING
        elif piece == self.BLACK and end_row == 7:
            self.board[end_row, end_col] = self.BLACK_KING

    def is_game_over(self):
        """Check if the game is over (no valid moves for current player)"""
        return len(self.get_valid_moves(self.current_player)) == 0

    def get_state(self):
        """Return current board state"""
        return self.board.copy() 