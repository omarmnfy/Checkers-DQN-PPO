import pygame
import numpy as np
from board import Board

class CheckersVisualizer:
    def __init__(self, window_size=600):
        pygame.init()
        self.window_size = window_size
        self.square_size = window_size // 8
        self.piece_radius = self.square_size // 2 - 10
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.DARK_RED = (139, 0, 0)
        self.GRAY = (128, 128, 128)
        self.BROWN = (139, 69, 19)
        self.BEIGE = (245, 245, 220)
        
        # Initialize display
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption('Checkers')

    def draw_board(self, state):
        self.screen.fill(self.WHITE)
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                x = col * self.square_size
                y = row * self.square_size
                color = self.BEIGE if (row + col) % 2 == 0 else self.BROWN
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))

        # Draw pieces
        for row in range(8):
            for col in range(8):
                x = col * self.square_size + self.square_size // 2
                y = row * self.square_size + self.square_size // 2
                piece = state[row][col]
                
                if piece != Board.EMPTY:
                    # Draw main piece circle
                    color = self.RED if piece in [Board.RED, Board.RED_KING] else self.BLACK
                    pygame.draw.circle(self.screen, color, (x, y), self.piece_radius)
                    
                    # Draw king indicator
                    if piece in [Board.RED_KING, Board.BLACK_KING]:
                        crown_color = self.DARK_RED if piece == Board.RED_KING else self.GRAY
                        pygame.draw.circle(self.screen, crown_color, (x, y), self.piece_radius // 2)

        pygame.display.flip()

    def close(self):
        pygame.quit() 