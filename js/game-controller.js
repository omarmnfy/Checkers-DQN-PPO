class GameController {
    constructor() {
        this.game = new CheckersGame();
        this.ai = new AIPlayer('medium');
        this.gameMode = 'human-vs-human';
        this.isAITurn = false;
        this.continueCapturing = false;
        this.capturingPiece = null;
        
        this.initializeDOM();
        this.bindEvents();
        this.renderBoard();
        this.updateGameInfo();
    }

    initializeDOM() {
        this.boardElement = document.getElementById('gameBoard');
        this.turnIndicator = document.getElementById('turnIndicator');
        this.gameStatus = document.getElementById('gameStatus');
        this.redCaptured = document.getElementById('redCaptured');
        this.blackCaptured = document.getElementById('blackCaptured');
        this.movesList = document.getElementById('movesList');
        this.gameOverModal = document.getElementById('gameOverModal');
        this.gameOverTitle = document.getElementById('gameOverTitle');
        this.gameOverMessage = document.getElementById('gameOverMessage');
        
        // Create board squares
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const square = document.createElement('div');
                square.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
                square.dataset.row = row;
                square.dataset.col = col;
                square.addEventListener('click', (e) => this.handleSquareClick(e));
                this.boardElement.appendChild(square);
            }
        }
    }

    bindEvents() {
        document.getElementById('newGameBtn').addEventListener('click', () => this.newGame());
        document.getElementById('undoBtn').addEventListener('click', () => this.undoMove());
        document.getElementById('gameMode').addEventListener('change', (e) => {
            this.gameMode = e.target.value;
            this.newGame();
        });
        document.getElementById('playAgainBtn').addEventListener('click', () => {
            this.hideGameOverModal();
            this.newGame();
        });
    }

    newGame() {
        this.game.reset();
        this.isAITurn = false;
        this.continueCapturing = false;
        this.capturingPiece = null;
        this.renderBoard();
        this.updateGameInfo();
        this.clearMoveHistory();
        this.hideGameOverModal();
        
        // If AI plays as red and it's human vs AI mode
        if (this.gameMode === 'human-vs-ai' && this.game.getCurrentPlayer() === CheckersGame.RED) {
            setTimeout(() => this.makeAIMove(), 500);
        }
    }

    handleSquareClick(event) {
        if (this.isAITurn || this.game.isGameOver()) return;

        const row = parseInt(event.target.dataset.row);
        const col = parseInt(event.target.dataset.col);
        
        if (isNaN(row) || isNaN(col)) return;

        const piece = this.game.board[row][col];
        const currentPlayer = this.game.getCurrentPlayer();

        // If we're in continue capturing mode, only allow moves with the capturing piece
        if (this.continueCapturing && this.capturingPiece) {
            if (this.game.selectedPiece && 
                this.game.selectedPiece.row === this.capturingPiece.row && 
                this.game.selectedPiece.col === this.capturingPiece.col) {
                // Try to make a move
                this.tryMove(row, col);
            } else if (row === this.capturingPiece.row && col === this.capturingPiece.col) {
                // Select the capturing piece
                this.selectPiece(row, col);
            }
            return;
        }

        // If clicking on own piece, select it
        if (this.game.isPiece(row, col, currentPlayer)) {
            this.selectPiece(row, col);
        }
        // If a piece is selected and clicking on empty square or opponent piece, try to move
        else if (this.game.selectedPiece) {
            this.tryMove(row, col);
        }
    }

    selectPiece(row, col) {
        const currentPlayer = this.game.getCurrentPlayer();
        
        if (!this.game.isPiece(row, col, currentPlayer)) return;

        this.game.selectedPiece = { row, col };
        
        // Get valid moves for this piece
        let validMoves;
        if (this.continueCapturing && this.capturingPiece) {
            // Only show captures for the capturing piece
            validMoves = this.game.getValidCaptures(row, col);
        } else {
            // Get all valid moves for current player, then filter for this piece
            const allMoves = this.game.getValidMoves(currentPlayer);
            validMoves = allMoves.filter(move => move.from.row === row && move.from.col === col);
        }
        
        this.game.validMoves = validMoves;
        this.renderBoard();
        
        if (validMoves.length === 0) {
            this.updateStatus("This piece has no valid moves.");
        } else if (this.game.mustCapture) {
            this.updateStatus("You must capture an opponent's piece.");
        } else {
            this.updateStatus("Click on a highlighted square to move.");
        }
    }

    tryMove(row, col) {
        if (!this.game.selectedPiece || !this.game.validMoves) return;

        // Find the move that matches the destination
        const move = this.game.validMoves.find(m => m.to.row === row && m.to.col === col);
        
        if (!move) {
            this.updateStatus("Invalid move. Please select a highlighted square.");
            return;
        }

        // Make the move
        const result = this.game.makeMove(move);
        this.addMoveToHistory(move);
        
        // Handle continue capturing
        if (result.continueCapturing) {
            this.continueCapturing = true;
            this.capturingPiece = result.piece;
            this.game.selectedPiece = result.piece;
            this.game.validMoves = this.game.getValidCaptures(result.piece.row, result.piece.col);
            this.updateStatus("You must continue capturing with the same piece!");
        } else {
            this.continueCapturing = false;
            this.capturingPiece = null;
            this.game.selectedPiece = null;
            this.game.validMoves = [];
        }

        this.renderBoard();
        this.updateGameInfo();

        // Check for game over
        if (this.game.isGameOver()) {
            this.handleGameOver();
            return;
        }

        // Handle AI turn
        if (this.gameMode === 'human-vs-ai' && !this.continueCapturing) {
            this.isAITurn = true;
            this.updateStatus("AI is thinking...");
            setTimeout(() => this.makeAIMove(), 1000);
        } else if (!this.continueCapturing) {
            this.updateStatus("Click on a piece to select it, then click on a valid square to move.");
        }
    }

    makeAIMove() {
        if (!this.isAITurn || this.game.isGameOver()) return;

        const currentPlayer = this.game.getCurrentPlayer();
        const move = this.ai.getBestMove(this.game, currentPlayer);

        if (!move) {
            this.handleGameOver();
            return;
        }

        const result = this.game.makeMove(move);
        this.addMoveToHistory(move);

        // Handle continue capturing for AI
        if (result.continueCapturing) {
            this.renderBoard();
            this.updateGameInfo();
            setTimeout(() => this.makeAIMove(), 800);
        } else {
            this.isAITurn = false;
            this.renderBoard();
            this.updateGameInfo();

            if (this.game.isGameOver()) {
                this.handleGameOver();
            } else {
                this.updateStatus("Your turn! Click on a piece to select it.");
            }
        }
    }

    undoMove() {
        if (this.isAITurn || this.continueCapturing) return;
        
        if (this.game.undoMove()) {
            this.renderBoard();
            this.updateGameInfo();
            this.updateMoveHistory();
            this.updateStatus("Move undone.");
        } else {
            this.updateStatus("No moves to undo.");
        }
    }

    renderBoard() {
        const squares = this.boardElement.children;
        const board = this.game.getBoard();

        for (let i = 0; i < squares.length; i++) {
            const square = squares[i];
            const row = parseInt(square.dataset.row);
            const col = parseInt(square.dataset.col);
            const piece = board[row][col];

            // Clear previous classes
            square.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
            square.innerHTML = '';

            // Add piece if present
            if (piece !== CheckersGame.EMPTY) {
                const pieceElement = document.createElement('div');
                pieceElement.className = 'piece';
                
                if (piece === CheckersGame.RED || piece === CheckersGame.RED_KING) {
                    pieceElement.classList.add('red');
                } else {
                    pieceElement.classList.add('black');
                }

                if (piece === CheckersGame.RED_KING || piece === CheckersGame.BLACK_KING) {
                    pieceElement.classList.add('king');
                }

                square.appendChild(pieceElement);
            }

            // Highlight selected piece
            if (this.game.selectedPiece && 
                this.game.selectedPiece.row === row && 
                this.game.selectedPiece.col === col) {
                square.classList.add('selected');
            }

            // Highlight valid moves
            if (this.game.validMoves) {
                const isValidMove = this.game.validMoves.some(move => 
                    move.to.row === row && move.to.col === col
                );
                if (isValidMove) {
                    square.classList.add('valid-move');
                }
            }

            // Highlight pieces that must capture
            if (this.game.mustCapture && !this.continueCapturing) {
                const currentPlayer = this.game.getCurrentPlayer();
                if (this.game.isPiece(row, col, currentPlayer)) {
                    const captures = this.game.getValidCaptures(row, col);
                    if (captures.length > 0) {
                        square.classList.add('must-capture');
                    }
                }
            }
        }
    }

    updateGameInfo() {
        const currentPlayer = this.game.getCurrentPlayer();
        const playerName = currentPlayer === CheckersGame.RED ? 'Red' : 'Black';
        
        this.turnIndicator.textContent = `${playerName}'s Turn`;
        
        // Update player indicators
        document.querySelectorAll('.player').forEach(player => {
            player.classList.remove('active');
        });
        
        if (currentPlayer === CheckersGame.RED) {
            document.querySelector('.red-player').classList.add('active');
        } else {
            document.querySelector('.black-player').classList.add('active');
        }

        // Update captured pieces
        this.updateCapturedPieces();
    }

    updateCapturedPieces() {
        const captured = this.game.getCapturedPieces();
        
        this.redCaptured.innerHTML = '';
        this.blackCaptured.innerHTML = '';

        captured.red.forEach(() => {
            const piece = document.createElement('div');
            piece.className = 'captured-piece red';
            this.redCaptured.appendChild(piece);
        });

        captured.black.forEach(() => {
            const piece = document.createElement('div');
            piece.className = 'captured-piece black';
            this.blackCaptured.appendChild(piece);
        });
    }

    addMoveToHistory(move) {
        const moveText = this.formatMove(move);
        const moveElement = document.createElement('div');
        moveElement.className = `move-item ${move.type === 'capture' ? 'capture' : ''}`;
        moveElement.textContent = `${this.game.gameHistory.length}. ${moveText}`;
        this.movesList.appendChild(moveElement);
        this.movesList.scrollTop = this.movesList.scrollHeight;
    }

    updateMoveHistory() {
        this.movesList.innerHTML = '';
        this.game.getGameHistory().forEach((historyItem, index) => {
            const moveElement = document.createElement('div');
            moveElement.className = `move-item ${historyItem.move.type === 'capture' ? 'capture' : ''}`;
            moveElement.textContent = `${index + 1}. ${this.formatMove(historyItem.move)}`;
            this.movesList.appendChild(moveElement);
        });
    }

    clearMoveHistory() {
        this.movesList.innerHTML = '';
    }

    formatMove(move) {
        const fromSquare = String.fromCharCode(97 + move.from.col) + (8 - move.from.row);
        const toSquare = String.fromCharCode(97 + move.to.col) + (8 - move.to.row);
        const symbol = move.type === 'capture' ? 'x' : '-';
        return `${fromSquare}${symbol}${toSquare}`;
    }

    updateStatus(message) {
        this.gameStatus.textContent = message;
    }

    handleGameOver() {
        const winner = this.game.getWinner();
        const winnerName = winner === CheckersGame.RED ? 'Red' : 'Black';
        
        this.gameOverTitle.textContent = 'Game Over!';
        this.gameOverMessage.textContent = `${winnerName} player wins!`;
        this.showGameOverModal();
        
        this.updateStatus(`Game Over! ${winnerName} wins!`);
    }

    showGameOverModal() {
        this.gameOverModal.classList.add('show');
    }

    hideGameOverModal() {
        this.gameOverModal.classList.remove('show');
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new GameController();
});