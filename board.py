import numpy as np

class Board:
    # Constants for piece representation
    EMPTY = 0
    RED = 1
    BLACK = 2
    RED_KING = 3
    BLACK_KING = 4

    def __init__(self):
        self.reset()

    def reset(self):
        """Initialize the board with pieces in starting positions"""
        self.board = np.zeros((8, 8), dtype=int)
        
        # Set up black pieces
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = self.BLACK

        # Set up red pieces
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = self.RED

    def get_valid_moves(self, color):
        """Returns all valid moves for the given color"""
        moves = []
        for row in range(8):
            for col in range(8):
                if self.board[row][col] in [color, color + 2]:  # Regular piece or king
                    # Check all possible moves for this piece
                    piece_moves = self._get_piece_moves(row, col)
                    moves.extend(piece_moves)
        return moves

    def _get_piece_moves(self, row, col):
        """Get all possible moves for a piece at the given position"""
        moves = []
        piece = self.board[row][col]
        
        if piece == 0:
            return moves

        directions = []
        # Regular pieces can only move in one direction
        if piece == self.RED or piece == self.RED_KING:
            directions.extend([(-1, -1), (-1, 1)])  # Up-left and up-right
        if piece == self.BLACK or piece == self.BLACK_KING:
            directions.extend([(1, -1), (1, 1)])    # Down-left and down-right
            
        # Check each direction
        for dy, dx in directions:
            new_row, new_col = row + dy, col + dx
            if self._is_valid_position(new_row, new_col):
                # Regular move
                if self.board[new_row][new_col] == self.EMPTY:
                    moves.append(((row, col), (new_row, new_col)))
                
                # Jump move
                elif self._can_jump(row, col, new_row, new_col):
                    jump_row, jump_col = new_row + dy, new_col + dx
                    moves.append(((row, col), (jump_row, jump_col)))

        return moves

    def _is_valid_position(self, row, col):
        """Check if the position is within the board"""
        return 0 <= row < 8 and 0 <= col < 8

    def _can_jump(self, start_row, start_col, mid_row, mid_col):
        """Check if a piece can jump over an opponent's piece"""
        if not self._is_valid_position(mid_row, mid_col):
            return False

        piece = self.board[start_row][start_col]
        target = self.board[mid_row][mid_col]
        
        # Calculate the landing position
        dy = mid_row - start_row
        dx = mid_col - start_col
        end_row, end_col = mid_row + dy, mid_col + dx

        # Check if landing position is valid and empty
        if not self._is_valid_position(end_row, end_col):
            return False
        if self.board[end_row][end_col] != self.EMPTY:
            return False

        # Check if jumping over opponent's piece
        if piece in [self.RED, self.RED_KING]:
            return target in [self.BLACK, self.BLACK_KING]
        else:
            return target in [self.RED, self.RED_KING]

    def make_move(self, start, end):
        """Make a move on the board"""
        start_row, start_col = start
        end_row, end_col = end
        
        # Move the piece
        piece = self.board[start_row][start_col]
        self.board[start_row][start_col] = self.EMPTY
        self.board[end_row][end_col] = piece

        # Handle jumps
        if abs(end_row - start_row) == 2:
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            self.board[mid_row][mid_col] = self.EMPTY

        # King promotion
        if piece == self.RED and end_row == 0:
            self.board[end_row][end_col] = self.RED_KING
        elif piece == self.BLACK and end_row == 7:
            self.board[end_row][end_col] = self.BLACK_KING

    def get_state(self):
        """Return the current state of the board"""
        return self.board.copy() 