import os
import random
# from model import TicTacToeModel
import copy
from tensorflow import keras
import tensorflow as tf
import pickle
from tqdm import tqdm

PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '
PLAYER_X_VAL = -1
PLAYER_O_VAL = 1
EMPTY_VAL = 0
HORIZONTAL_SEPARATOR = ' | '
VERTICAL_SEPARATOR = '---------------'
GAME_STATE_X = -1
GAME_STATE_O = 1
GAME_STATE_DRAW = 0
GAME_STATE_NOT_ENDED = 2
ALL_GAMES_PATH = '/content/drive/MyDrive/18.0651/tic_net/all_games.pkl'

class Game:

    def __init__(self):
        self.resetBoard()
        self.trainingHistory = []

    def resetBoard(self):
        self.board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.boardHistory = []

    def printBoard(self):
        print(VERTICAL_SEPARATOR)
        for i in range(len(self.board)):
            print(' ', end='')
            for j in range(len(self.board[i])):
                if PLAYER_X_VAL == self.board[i][j]:
                    print(PLAYER_X, end='')
                elif PLAYER_O_VAL == self.board[i][j]:
                    print(PLAYER_O, end='')
                elif EMPTY_VAL == self.board[i][j]:
                    print(EMPTY, end='')
                print(HORIZONTAL_SEPARATOR, end='')
            print(os.linesep)
            print(VERTICAL_SEPARATOR)

    def getGameResult(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == EMPTY_VAL:
                    return GAME_STATE_NOT_ENDED

        # Rows
        for i in range(len(self.board)):
            candidate = self.board[i][0]
            for j in range(len(self.board[i])):
                if candidate != self.board[i][j]:
                    candidate = 0
            if candidate != 0:
                return candidate

        # Columns
        for i in range(len(self.board)):
            candidate = self.board[0][i]
            for j in range(len(self.board[i])):
                if candidate != self.board[j][i]:
                    candidate = 0
            if candidate != 0:
                return candidate

        # First diagonal
        candidate = self.board[0][0]
        for i in range(len(self.board)):
            if candidate != self.board[i][i]:
                candidate = 0
        if candidate != 0:
            return candidate

        # Second diagonal
        candidate = self.board[0][2]
        for i in range(len(self.board)):
            if candidate != self.board[i][len(self.board[i]) - i - 1]:
                candidate = 0
        if candidate != 0:
            return candidate

        return GAME_STATE_DRAW


    def getAvailableMoves(self):
        availableMoves = []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if (self.board[i][j]) == EMPTY_VAL:
                    availableMoves.append([i, j])
        return availableMoves

    def addToHistory(self, board):
        self.boardHistory.append(board)

    def printHistory(self):
        print(self.boardHistory)

    def displayBoard(self):
        row_num = 0
        for row in self.board:
            row_num += 1
            row_string = ""

            for square in row:
                if square == PLAYER_X_VAL:
                    row_string += PLAYER_X
                elif square == PLAYER_O_VAL:
                    row_string += PLAYER_O
                else:
                    row_string += " "
                row_string += " | "
            print(row_string[:-3])
            if row_num < 3:
                print("―"*9)

    def move(self, position, player):
        availableMoves = self.getAvailableMoves()
        for i in range(len(availableMoves)):
            if position[0] == availableMoves[i][0] and position[1] == availableMoves[i][1]:
                self.board[position[0]][position[1]] = player
                self.addToHistory(copy.deepcopy(self.board))


    def simulate(self, playerToMove):
        while (self.getGameResult() == GAME_STATE_NOT_ENDED):
            availableMoves = self.getAvailableMoves()
            selectedMove = availableMoves[random.randrange(0, len(availableMoves))]
            self.move(selectedMove, playerToMove)
            if playerToMove == PLAYER_X_VAL:
                playerToMove = PLAYER_O_VAL
            else:
                playerToMove = PLAYER_X_VAL
        # Get the history and build the training set
        for historyItem in self.boardHistory:
            self.trainingHistory.append((self.getGameResult(), copy.deepcopy(historyItem)))

    def simulateNeuralNetwork(self, nnPlayer, model):
        playerToMove = PLAYER_X_VAL
        while (self.getGameResult() == GAME_STATE_NOT_ENDED):
            availableMoves = self.getAvailableMoves()
            if playerToMove == nnPlayer:
                maxValue = 0
                bestMove = availableMoves[0]
                for availableMove in availableMoves:
                    # get a copy of a board
                    boardCopy = copy.deepcopy(self.board)
                    boardCopy[availableMove[0]][availableMove[1]] = nnPlayer
                    if nnPlayer == PLAYER_X_VAL:
                        value = model.predict(boardCopy, 0)
                    else:
                        value = model.predict(boardCopy, 2)
                    if value > maxValue:
                        maxValue = value
                        bestMove = availableMove
                selectedMove = bestMove
            else:
                selectedMove = availableMoves[random.randrange(0, len(availableMoves))]
            self.move(selectedMove, playerToMove)
            if playerToMove == PLAYER_X_VAL:
                playerToMove = PLAYER_O_VAL
            else:
                playerToMove = PLAYER_X_VAL

    def getTrainingHistory(self):
        return self.trainingHistory

    def simulateManyGames(self, playerToMove, numberOfGames):
        playerXWins = 0
        playerOWins = 0
        draws = 0
        for i in range(numberOfGames):
            self.resetBoard()
            self.simulate(playerToMove)
            if self.getGameResult() == PLAYER_X_VAL:
                playerXWins = playerXWins + 1
            elif self.getGameResult() == PLAYER_O_VAL:
                playerOWins = playerOWins + 1
            else: draws = draws + 1
        totalWins = playerXWins + playerOWins + draws
        print ('X Wins: ' + str(int(playerXWins * 100/totalWins)) + '%')
        print('O Wins: ' + str(int(playerOWins * 100 / totalWins)) + '%')
        print('Draws: ' + str(int(draws * 100 / totalWins)) + '%')


    def simulateManyNeuralNetworkGames(self, nnPlayer, numberOfGames, model, updateModel = True):
        nnPlayerWins = 0
        randomPlayerWins = 0
        draws = 0
        print ("NN player")
        print (nnPlayer)
        for i in tqdm(range(numberOfGames)):
            self.resetBoard()
            self.simulateNeuralNetwork(nnPlayer, model)
            if self.getGameResult() == nnPlayer:
                nnPlayerWins = nnPlayerWins + 1
            elif self.getGameResult() == GAME_STATE_DRAW:
                draws = draws + 1
            else: 
                randomPlayerWins = randomPlayerWins + 1
                # model.updateWithOne(copy.deepcopy(self.board))
        totalWins = nnPlayerWins + randomPlayerWins + draws
        # print ('X Wins: ' + str(int(nnPlayerWins * 100/totalWins)) + '%')
        # print('O Wins: ' + str(int(randomPlayerWins * 100 / totalWins)) + '%')
        # print('Draws: ' + str(int(draws * 100 / totalWins)) + '%')
        print('X Wins: ' + str(nnPlayerWins))
        print('O Wins: ' + str(randomPlayerWins))
        print('Draws: ' + str(draws))

    def playWithMe(self, nnPlayer, model):
        playerToMove = PLAYER_X_VAL
        while (self.getGameResult() == GAME_STATE_NOT_ENDED):
            availableMoves = self.getAvailableMoves()
            if playerToMove == nnPlayer:
                maxValue = 0
                bestMove = availableMoves[0]
                for availableMove in availableMoves:
                    # get a copy of a board
                    boardCopy = copy.deepcopy(self.board)
                    boardCopy[availableMove[0]][availableMove[1]] = nnPlayer
                    if nnPlayer == PLAYER_X_VAL:
                        value = model.predict(boardCopy, 0)
                    else:
                        value = model.predict(boardCopy, 2)
                    if value > maxValue:
                        maxValue = value
                        bestMove = availableMove
                selectedMove = bestMove
            else:
                selectedMove = None
                print('current board')
                self.displayBoard()
                print("here are the available moves: ", [[z+1 for z in y] for y in availableMoves])
                while selectedMove not in availableMoves:
                    try:
                        row, col = input("your turn:  ").split(" ")
                        row = int(row) - 1
                        col = int(col) - 1
                        selectedMove = [row, col]
                        print(selectedMove)
                    except:
                        print("oops")

            self.move(selectedMove, playerToMove)
            if playerToMove == PLAYER_X_VAL:
                playerToMove = PLAYER_O_VAL
            else:
                playerToMove = PLAYER_X_VAL
        print(self.getGameResult())

ALL_GAMES = {}
def simulateAllGames(current_game, playerToMove):
    result = current_game.getGameResult()
    if result != GAME_STATE_NOT_ENDED:

        return result == PLAYER_X_VAL, result == PLAYER_O_VAL, result == GAME_STATE_DRAW
    else:
        next_player = PLAYER_X_VAL if playerToMove == PLAYER_O_VAL else PLAYER_O_VAL
        win_rate = (0, 0, 0)
        for move in current_game.getAvailableMoves():
            new_game = Game()
            new_game.board = copy.deepcopy(current_game.board)
            new_game.move(move, playerToMove)
            new_win_rate = simulateAllGames(new_game, next_player)
            board_key = tuple(map(tuple, new_game.board))
            if board_key in ALL_GAMES:
                ALL_GAMES[board_key] = tuple(map(sum, zip(ALL_GAMES[board_key], new_win_rate)))
            else:
                if len(ALL_GAMES) % 1000 == 0:
                    print(len(ALL_GAMES), "/", 8952)
                ALL_GAMES[board_key] = new_win_rate
            win_rate = tuple(map(sum, zip(win_rate, new_win_rate)))
        return win_rate

def generate_all_games(data_path):
    game = Game()
    simulateAllGames(game, PLAYER_X_VAL)
    simulateAllGames(game, PLAYER_O_VAL)
    all_game_states = []
    for board, win_rate in ALL_GAMES.items():
        board_as_list = [list(row) for row in board]
        all_game_states.append((tuple(float(i)/sum(win_rate) for i in win_rate), board_as_list))
    with open(data_path, 'wb') as f:
        pickle.dump(all_game_states, f)
        print(len(all_game_states))

def load_all_games(data_path):
    with open(data_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    print(tf.__version__)
    game = Game()
    # generate_all_games()
    all_game_states = load_all_games()
    print("ALL", all_game_states[0])
    model_path = '/content/drive/MyDrive/18.0651/tic_net/model.h5'
    ticTacToeModel = TicTacToeModel(9, 3, 100, 32)
    train_model = False
    if train_model:
        ticTacToeModel.train(all_game_states)
        ticTacToeModel.save_weights(model_path)
        print("Saved model to ", model_path)
    else:
        ticTacToeModel.built = True
        print("Loading model from ", model_path)
        ticTacToeModel.load_weights(model_path) #pickle.load(filehandler)

    print ("Simulating with Neural Network as X Player:")
    game.simulateManyNeuralNetworkGames(PLAYER_X_VAL, 1000, ticTacToeModel)
    print("Simulating with Neural Network as O Player:")
    game.simulateManyNeuralNetworkGames(PLAYER_O_VAL, 1000, ticTacToeModel)
    # game.playWithMe(PLAYER_X_VAL, ticTacToeModel)
