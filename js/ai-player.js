class AIPlayer {
    constructor(difficulty = 'medium') {
        this.difficulty = difficulty;
        this.maxDepth = this.getMaxDepth(difficulty);
    }

    getMaxDepth(difficulty) {
        switch (difficulty) {
            case 'easy': return 2;
            case 'medium': return 4;
            case 'hard': return 6;
            default: return 4;
        }
    }

    getBestMove(game, player) {
        const validMoves = game.getValidMoves(player);
        if (validMoves.length === 0) return null;

        if (this.difficulty === 'easy' && Math.random() < 0.3) {
            // 30% chance of random move for easy difficulty
            return validMoves[Math.floor(Math.random() * validMoves.length)];
        }

        const result = this.minimax(game, this.maxDepth, -Infinity, Infinity, true, player);
        return result.move;
    }

    minimax(game, depth, alpha, beta, maximizingPlayer, originalPlayer) {
        if (depth === 0 || game.isGameOver()) {
            return {
                score: this.evaluateBoard(game, originalPlayer),
                move: null
            };
        }

        const currentPlayer = game.getCurrentPlayer();
        const validMoves = game.getValidMoves(currentPlayer);

        if (validMoves.length === 0) {
            return {
                score: maximizingPlayer ? -Infinity : Infinity,
                move: null
            };
        }

        let bestMove = null;

        if (maximizingPlayer) {
            let maxScore = -Infinity;
            
            for (const move of validMoves) {
                const gameClone = this.cloneGame(game);
                const result = gameClone.makeMove(move);
                
                let score;
                if (result.continueCapturing) {
                    // Same player continues, but reduce depth
                    score = this.minimax(gameClone, depth - 1, alpha, beta, true, originalPlayer).score;
                } else {
                    // Switch to minimizing player
                    score = this.minimax(gameClone, depth - 1, alpha, beta, false, originalPlayer).score;
                }

                if (score > maxScore) {
                    maxScore = score;
                    bestMove = move;
                }

                alpha = Math.max(alpha, score);
                if (beta <= alpha) {
                    break; // Alpha-beta pruning
                }
            }

            return { score: maxScore, move: bestMove };
        } else {
            let minScore = Infinity;
            
            for (const move of validMoves) {
                const gameClone = this.cloneGame(game);
                const result = gameClone.makeMove(move);
                
                let score;
                if (result.continueCapturing) {
                    // Same player continues, but reduce depth
                    score = this.minimax(gameClone, depth - 1, alpha, beta, false, originalPlayer).score;
                } else {
                    // Switch to maximizing player
                    score = this.minimax(gameClone, depth - 1, alpha, beta, true, originalPlayer).score;
                }

                if (score < minScore) {
                    minScore = score;
                    bestMove = move;
                }

                beta = Math.min(beta, score);
                if (beta <= alpha) {
                    break; // Alpha-beta pruning
                }
            }

            return { score: minScore, move: bestMove };
        }
    }

    evaluateBoard(game, player) {
        const opponent = player === CheckersGame.RED ? CheckersGame.BLACK : CheckersGame.RED;
        
        // Check for game over conditions
        if (game.isGameOver()) {
            const winner = game.getWinner();
            if (winner === player) return 1000;
            if (winner === opponent) return -1000;
            return 0; // Draw
        }

        let score = 0;
        const board = game.getBoard();

        // Count pieces and evaluate positions
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = board[row][col];
                
                if (piece === CheckersGame.EMPTY) continue;

                let pieceValue = 0;
                let isPlayerPiece = false;

                // Basic piece values
                if (piece === player) {
                    pieceValue = 10;
                    isPlayerPiece = true;
                } else if (piece === player + 2) { // King
                    pieceValue = 15;
                    isPlayerPiece = true;
                } else if (piece === opponent) {
                    pieceValue = -10;
                } else if (piece === opponent + 2) { // King
                    pieceValue = -15;
                }

                // Position bonuses
                if (isPlayerPiece) {
                    // Encourage advancement
                    if (player === CheckersGame.RED) {
                        pieceValue += (7 - row) * 0.5; // Closer to top is better for red
                    } else {
                        pieceValue += row * 0.5; // Closer to bottom is better for black
                    }

                    // Center control bonus
                    if (col >= 2 && col <= 5) {
                        pieceValue += 1;
                    }

                    // Edge penalty
                    if (col === 0 || col === 7) {
                        pieceValue -= 1;
                    }
                } else {
                    // Apply negative bonuses for opponent pieces
                    if (opponent === CheckersGame.RED) {
                        pieceValue -= (7 - row) * 0.5;
                    } else {
                        pieceValue -= row * 0.5;
                    }

                    if (col >= 2 && col <= 5) {
                        pieceValue -= 1;
                    }

                    if (col === 0 || col === 7) {
                        pieceValue += 1;
                    }
                }

                score += pieceValue;
            }
        }

        // Mobility bonus - having more moves is better
        const playerMoves = game.getValidMoves(player).length;
        const opponentMoves = game.getValidMoves(opponent).length;
        score += (playerMoves - opponentMoves) * 0.5;

        return score;
    }

    cloneGame(game) {
        const clone = new CheckersGame();
        clone.board = game.getBoard();
        clone.currentPlayer = game.getCurrentPlayer();
        clone.capturedPieces = game.getCapturedPieces();
        clone.gameHistory = game.getGameHistory();
        return clone;
    }
}