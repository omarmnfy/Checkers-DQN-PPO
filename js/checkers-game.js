class CheckersGame {
    static EMPTY = 0;
    static RED = 1;
    static BLACK = 2;
    static RED_KING = 3;
    static BLACK_KING = 4;

    constructor() {
        this.board = null;
        this.currentPlayer = null;
        this.selectedPiece = null;
        this.validMoves = [];
        this.mustCapture = false;
        this.gameHistory = [];
        this.capturedPieces = { red: [], black: [] };
        this.reset();
    }

    reset() {
        this.board = Array(8).fill().map(() => Array(8).fill(CheckersGame.EMPTY));
        this.currentPlayer = CheckersGame.RED;
        this.selectedPiece = null;
        this.validMoves = [];
        this.mustCapture = false;
        this.gameHistory = [];
        this.capturedPieces = { red: [], black: [] };
        
        // Place black pieces (top of board)
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 8; col++) {
                if ((row + col) % 2 === 1) {
                    this.board[row][col] = CheckersGame.BLACK;
                }
            }
        }

        // Place red pieces (bottom of board)
        for (let row = 5; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                if ((row + col) % 2 === 1) {
                    this.board[row][col] = CheckersGame.RED;
                }
            }
        }
    }

    isValidPosition(row, col) {
        return row >= 0 && row < 8 && col >= 0 && col < 8;
    }

    isPiece(row, col, player) {
        if (!this.isValidPosition(row, col)) return false;
        const piece = this.board[row][col];
        return piece === player || piece === player + 2; // Regular piece or king
    }

    isOpponentPiece(row, col, player) {
        if (!this.isValidPosition(row, col)) return false;
        const opponent = player === CheckersGame.RED ? CheckersGame.BLACK : CheckersGame.RED;
        return this.isPiece(row, col, opponent);
    }

    getValidMoves(player) {
        const moves = [];
        const captures = [];

        // First, check for captures (mandatory in checkers)
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                if (this.isPiece(row, col, player)) {
                    const pieceCaptures = this.getValidCaptures(row, col);
                    captures.push(...pieceCaptures);
                }
            }
        }

        // If captures are available, only return captures
        if (captures.length > 0) {
            this.mustCapture = true;
            return captures;
        }

        // Otherwise, get regular moves
        this.mustCapture = false;
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                if (this.isPiece(row, col, player)) {
                    const pieceMoves = this.getValidRegularMoves(row, col);
                    moves.push(...pieceMoves);
                }
            }
        }

        return moves;
    }

    getValidCaptures(row, col) {
        const captures = [];
        const piece = this.board[row][col];
        const directions = this.getMoveDirections(piece);

        for (const [dr, dc] of directions) {
            const jumpRow = row + dr * 2;
            const jumpCol = col + dc * 2;
            const middleRow = row + dr;
            const middleCol = col + dc;

            if (this.isValidPosition(jumpRow, jumpCol) &&
                this.board[jumpRow][jumpCol] === CheckersGame.EMPTY &&
                this.isOpponentPiece(middleRow, middleCol, piece === CheckersGame.RED_KING ? CheckersGame.RED : 
                                   piece === CheckersGame.BLACK_KING ? CheckersGame.BLACK : piece)) {
                captures.push({
                    from: { row, col },
                    to: { row: jumpRow, col: jumpCol },
                    captured: { row: middleRow, col: middleCol },
                    type: 'capture'
                });
            }
        }

        return captures;
    }

    getValidRegularMoves(row, col) {
        const moves = [];
        const piece = this.board[row][col];
        const directions = this.getMoveDirections(piece);

        for (const [dr, dc] of directions) {
            const newRow = row + dr;
            const newCol = col + dc;

            if (this.isValidPosition(newRow, newCol) &&
                this.board[newRow][newCol] === CheckersGame.EMPTY) {
                moves.push({
                    from: { row, col },
                    to: { row: newRow, col: newCol },
                    type: 'move'
                });
            }
        }

        return moves;
    }

    getMoveDirections(piece) {
        const directions = [];
        
        if (piece === CheckersGame.RED) {
            directions.push([-1, -1], [-1, 1]); // Up-left, up-right
        } else if (piece === CheckersGame.BLACK) {
            directions.push([1, -1], [1, 1]); // Down-left, down-right
        } else if (piece === CheckersGame.RED_KING || piece === CheckersGame.BLACK_KING) {
            directions.push([-1, -1], [-1, 1], [1, -1], [1, 1]); // All directions
        }

        return directions;
    }

    makeMove(move) {
        const { from, to, captured, type } = move;
        const piece = this.board[from.row][from.col];
        
        // Save move to history
        this.gameHistory.push({
            move: { ...move },
            boardState: this.board.map(row => [...row]),
            currentPlayer: this.currentPlayer,
            capturedPieces: {
                red: [...this.capturedPieces.red],
                black: [...this.capturedPieces.black]
            }
        });

        // Move the piece
        this.board[from.row][from.col] = CheckersGame.EMPTY;
        this.board[to.row][to.col] = piece;

        // Handle capture
        if (type === 'capture') {
            const capturedPiece = this.board[captured.row][captured.col];
            this.board[captured.row][captured.col] = CheckersGame.EMPTY;
            
            // Add to captured pieces
            if (capturedPiece === CheckersGame.RED || capturedPiece === CheckersGame.RED_KING) {
                this.capturedPieces.red.push(capturedPiece);
            } else {
                this.capturedPieces.black.push(capturedPiece);
            }
        }

        // Handle king promotion
        if (piece === CheckersGame.RED && to.row === 0) {
            this.board[to.row][to.col] = CheckersGame.RED_KING;
        } else if (piece === CheckersGame.BLACK && to.row === 7) {
            this.board[to.row][to.col] = CheckersGame.BLACK_KING;
        }

        // Check for additional captures after a capture move
        if (type === 'capture') {
            const additionalCaptures = this.getValidCaptures(to.row, to.col);
            if (additionalCaptures.length > 0) {
                // Player must continue capturing with the same piece
                return { continueCapturing: true, piece: { row: to.row, col: to.col } };
            }
        }

        // Switch players
        this.currentPlayer = this.currentPlayer === CheckersGame.RED ? CheckersGame.BLACK : CheckersGame.RED;
        return { continueCapturing: false };
    }

    undoMove() {
        if (this.gameHistory.length === 0) return false;

        const lastState = this.gameHistory.pop();
        this.board = lastState.boardState;
        this.currentPlayer = lastState.currentPlayer;
        this.capturedPieces = lastState.capturedPieces;
        this.selectedPiece = null;
        this.validMoves = [];
        this.mustCapture = false;

        return true;
    }

    isGameOver() {
        const redMoves = this.getValidMoves(CheckersGame.RED);
        const blackMoves = this.getValidMoves(CheckersGame.BLACK);
        
        return redMoves.length === 0 || blackMoves.length === 0 ||
               this.countPieces(CheckersGame.RED) === 0 || this.countPieces(CheckersGame.BLACK) === 0;
    }

    getWinner() {
        if (!this.isGameOver()) return null;

        const redPieces = this.countPieces(CheckersGame.RED);
        const blackPieces = this.countPieces(CheckersGame.BLACK);
        const redMoves = this.getValidMoves(CheckersGame.RED);
        const blackMoves = this.getValidMoves(CheckersGame.BLACK);

        if (redPieces === 0 || redMoves.length === 0) {
            return CheckersGame.BLACK;
        } else if (blackPieces === 0 || blackMoves.length === 0) {
            return CheckersGame.RED;
        }

        return null; // Draw (shouldn't happen in checkers)
    }

    countPieces(player) {
        let count = 0;
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                if (this.isPiece(row, col, player)) {
                    count++;
                }
            }
        }
        return count;
    }

    getBoard() {
        return this.board.map(row => [...row]);
    }

    getCurrentPlayer() {
        return this.currentPlayer;
    }

    getCapturedPieces() {
        return { ...this.capturedPieces };
    }

    getGameHistory() {
        return [...this.gameHistory];
    }
}