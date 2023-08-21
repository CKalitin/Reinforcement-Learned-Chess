import Engine as E
import Model
import numpy as np
from collections import deque
import random
import ConstantData
import time
import numpy as np
import torch
import math
import datetime
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001 # Learning Rate

class Agent:
    # Config
    transitionMoves = 40 # Transition between START and END in ConstantData.py
    initialRandomness = 0.75
    numRandomMoves = 1000
    preivousWhiteLostPiece = '.'
    preivoueBlackLostPiece = '.'
    useRandomMoves = True
    movesToTry = 3888
    gamma = 0.9 # Discount rate
        
    def __init__(self):
        self.engine = E.Engine()
        self.numGames = 0
        self.epsilon = 0 # Randomness
        self.totalNumMoves = 0
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() on max memory
        self.model = Model.Linear_QNET(768, 2048, 8192, 3888) # Making up numbers
        self.trainer = Model.QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)
        self.model.loadModel()
        
    def Loop(self):
        self.engine.BeginGame()
        self.preivousWhiteLostPiece = '.'
        self.preivoueBlackLostPiece = '.'
        while(True):
            print(self.engine.board)
            
            isWhite = len(self.engine.board.move_stack) % 2 == 0
            
            move=input("Move: ")
            reward = self.RewardFunction(move, isWhite)
            self.engine.PlayMove(move)
            self.GetState(isWhite)
            
            print("Reward: ", reward)
            
    def GetState(self, whiteOnTop = True):
        state = [0] * 768
        
        # First 64 are pawns, then Knights, then Bishops, then Rooks, then Queens, then Kings
        # First 384 are white, next 384 are black
        # whiteOnTop defines if the white pieces are in the first or second half of the state array
        
        # Remove spaces
        boardString = str(self.engine.board).replace(" ", "").replace("\n", "")
        iters = 0
        for x in str(boardString):
            if x == ".":
                iters += 1
                continue
            
            letter = x.lower()
            index = 0
            
            if letter == "p": index = iters
            elif letter == "n": index = iters + 64
            elif letter == "b": index = iters + 128
            elif letter == "r": index = iters + 192
            elif letter == "q": index = iters + 256
            elif letter == "k": index = iters + 320
            
            if whiteOnTop and letter == x: index += 384 # If is lowercase ie. black
            if not whiteOnTop and letter.upper() == x: index += 384 # If is uppercase ie. white
            state[index] = 1
            iters += 1
        
        #PrintState(state)
        
        return np.array(state, dtype=int)
    
    def GetModelOutput(self, state):
        if self.useRandomMoves:
            self.epsilon = self.initialRandomness * np.clip(abs(self.totalNumMoves / self.numRandomMoves - 1), 0, 1)
            modelOutput = [0] * self.model.linear3.out_features
            if (random.randint(0, 100) / 100 > self.epsilon):
                state0 = torch.tensor(state, dtype=torch.float)
                modelOutput = self.model(state0)
            else: 
                for x in range(0, self.movesToTry):
                    modelOutput[random.randint(0, self.model.linear3.out_features - 1)] = 1 - (x / self.movesToTry)
                modelOutput = torch.tensor(modelOutput, dtype=torch.float)
            return modelOutput
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            modelOutput = self.model(state0)
        return modelOutput
    
    def PlayMove(self, modelOutput, isWhite):
        modelOutput = (torch.sort(modelOutput, descending=True)).indices # Sort and get the indices
        
        reward = 0
        termination = 0
        iters = 0
        outputMove = 0
        while iters < self.movesToTry:
            move = ConstantData.MODEL_OUTPUT_MOVES[modelOutput[iters]]
            outputMove = move
            if not isWhite: FlipMoveOnBoard(move) # If not white's turn, flip the move
            #print("move: ", move, modelOutput[iters], iters)
            reward = self.RewardFunction(move, isWhite)
            termination = self.engine.PlayMove(move)
            if (termination >= 0):
                if len(self.engine.board.move_stack) % 20 == 0:
                    print(f"Move: {move}\tIters: {iters}\tMove Number: {math.ceil(len(self.engine.board.move_stack) / 2)}")
                break
            iters += 1
            if iters >= self.movesToTry: 
                print(f"No valid moves found.\n{modelOutput}")
                return "error"
        return reward, termination, iters, outputMove
    
    # Run before the move executed, the old board is needed to get the piece used and the piece at the target square
    # Using FlipPositionOnBoard() is necessary because the board is flipped for display purposes, (black is on top in the console)
    def RewardFunction(self, move, isWhite):
        reward = 0
        boardString = str(self.engine.board).replace(" ", "").replace("\n", "")
        initialSquare = PositionToIndex(FlipPositionOnBoard(move[0:2]))
        piece = boardString[initialSquare].lower()
        targetSquare = PositionToIndex(FlipPositionOnBoard(move[2:4])) 
        targetPiece = boardString[targetSquare]
        
        # Piece position reward
        # Transition between start rewards and end rewards, (initReward / transitionMoves * (transitionMoves - moves)) + (endReward / transitionMoves * moves) = reward
        if not isWhite: targetSquare = PositionToIndex(move[2:4]) # Unflip the target square for black
        if piece == "p": reward += (ConstantData.PAWN_START_REWARD[targetSquare] / self.transitionMoves * (self.transitionMoves - (len(self.engine.board.move_stack) / 2))) + (ConstantData.PAWN_END_REWARD[targetSquare] / self.transitionMoves * (len(self.engine.board.move_stack) / 2))
        elif piece == "n": reward += ConstantData.KNIGHT_REWARD[targetSquare]
        elif piece == "b": reward += ConstantData.BISHOP_REWARD[targetSquare]
        elif piece == "r": reward += ConstantData.ROOK_REWARD[targetSquare]
        elif piece == "q": reward += ConstantData.QUEEN_REWARD[targetSquare]
        elif piece == "k": reward += (ConstantData.KING_START_REWARD[targetSquare] / self.transitionMoves * (self.transitionMoves - (len(self.engine.board.move_stack) / 2))) + (ConstantData.KING_START_REWARD[targetSquare] / self.transitionMoves * (len(self.engine.board.move_stack) / 2))
        
        # Promotion reward
        if len(move) > 4:
            if move[4] in ConstantData.PIECE_VALUE: reward += ConstantData.PIECE_VALUE[move[4]]
        
        # Losing/Taking pieces
        if isWhite: 
            self.preivoueBlackLostPiece = targetPiece.lower()
            reward -= ConstantData.PIECE_VALUE[self.preivousWhiteLostPiece]
            if targetPiece in ConstantData.BLACK_PIECES: reward += ConstantData.PIECE_VALUE[targetPiece]
        else: 
            self.preivousWhiteLostPiece = targetPiece.lower()
            reward -= ConstantData.PIECE_VALUE[self.preivoueBlackLostPiece]
            if targetPiece in ConstantData.WHITE_PIECES: reward += ConstantData.PIECE_VALUE[targetPiece.lower()]
        
        return reward
    
    # this is stupid and not how you do reward, just take the target square of a move and the piece used
    
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done)) # Adds tuple to memory
    
    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE: miniSample = random.sample(self.memory, BATCH_SIZE) # Returns list of tuples
        else: miniSample = self.memory
        
        states, actions, rewards, nextStates, dones = zip(*miniSample)
        
        self.trainer.trainStep(states, actions, rewards, nextStates, dones)
    
    def trainShortMemory(self, state, action, reward, nextState, done):
        self.trainer.trainStep(state, action, reward, nextState, done)
    
    def BeginGame(self):
        self.engine.BeginGame()
        self.preivousWhiteLostPiece = '.'
        self.preivoueBlackLostPiece = '.'
    
    
def PrintState(state):
    output = ""
    for x in range(0, int(len(state) / 8)):
        if x % 8 == 0: output += "\n"
        x *= 8
        output += f'\n{state[x]} {state[x + 1]} {state[x + 2]} {state[x + 3]} {state[x + 4]} {state[x + 5]} {state[x + 6]} {state[x + 7]}'
    output += "\n"
    print(output)

def FlipMoveOnBoard(move):
    # The model might be playing as black, so the moves need to be flipped
    outputMove = move[0] + str(9 - int(move[1])) + move[2] + str(9 - int(move[3]))
    return outputMove

def FlipPositionOnBoard(position):
    outputPosition = position[0] + str(9 - int(position[1]))
    return outputPosition

def PositionToIndex(position):
    letterToIndex = { 'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7 }
    return (int(position[1]) - 1) * 8 + letterToIndex[position[0]]

def Train():
    agent = Agent()
    agent.BeginGame()
    
    while(True):
        time.sleep(0.1)
            
        isWhite = len(agent.engine.board.move_stack) % 2 == 0
        
        initialState = agent.GetState(isWhite)
        modelOutput = agent.GetModelOutput(initialState)
        reward, termination, iters, move = agent.PlayMove(modelOutput, isWhite)
        if reward == "error": 
            print("Error")
            continue
        agent.totalNumMoves += 1
        newState = agent.GetState(isWhite)
        done = termination > 0
        if termination == 1: reward += 2000 # checkmate
        elif termination == 2: reward -= 1000 # stalemate
        elif termination == 3: reward -= 1000 # insufficient_material
        elif termination == 5: reward -= 1000 # is_fivefold_repitition
        moveIndex = ConstantData.MODEL_OUTPUT_MOVES.index(move)
        
        agent.trainShortMemory(initialState, moveIndex, reward, newState, done)
        agent.remember(initialState, moveIndex, reward, newState, done)
        if done:
            agent.numGames += 1
            agent.trainLongMemory()
            
            #agent.model.save(f"Model_{agent.engine.gameName[14:]}.pth")
            agent.model.save(f"Model.pth")
            
            print(f"Game {agent.numGames}, Ended in: {math.ceil(len(agent.engine.board.move_stack) / 2)} moves, Termination: {termination}")
            
            agent.engine.SaveGame()
            agent.BeginGame()

def PlayAgainst():
    agent = Agent()
    agent.BeginGame()
    
    print(agent.engine.board)
    while(True):
        isWhite = len(agent.engine.board.move_stack) % 2 == 0
        
        moveFound = False
        while not moveFound:
            move=input("Move: ")
            if agent.engine.PlayMove(move) >= 0: 
                agent.totalNumMoves += 1
                moveFound = True
        
        initialState = agent.GetState(isWhite)
        modelOutput = agent.GetModelOutput(initialState)
        reward, termination, iters, move = agent.PlayMove(modelOutput, isWhite)
        print("AI Move: " + move)
        if reward == "error": 
            print("Error")
            continue
        agent.totalNumMoves += 1
        
        print(agent.engine.board)
        

if __name__ == '__main__':
    #agent = Agent()
    #agent.Loop() # Play Manually
    Train()
    #PlayAgainst()