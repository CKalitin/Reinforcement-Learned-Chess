import Engine as E
import Model
import numpy as np
from collections import deque
import random
import ConstantData

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001 # Learning Rate

class Agent:
    def __init__(self):
        self.engine = E.Engine()
        self.transitionMoves = 40 # Transition between START and END in ConstantData.py
        self.numGames = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() on max memory
        self.model = Model.Linear_QNET(768, 2048, 8192, 3888) # Making up numbers
        self.trainer = Model.QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)
        
    def Loop(self):
        self.engine.BeginGame()
        while(True):
            print(self.engine.board)
            move=input("Move: ")
            self.engine.PlayMove(move)
            self.GetState(True)
            
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
            if letter == "n": index = iters + 64
            if letter == "b": index = iters + 128
            if letter == "r": index = iters + 192
            if letter == "q": index = iters + 256
            if letter == "k": index = iters + 320
            
            if whiteOnTop and letter == x: index += 384 # If is lowercase ie. black
            if not whiteOnTop and letter.upper() == x: index += 384 # If is uppercase ie. white
            state[index] = 1
            iters += 1
        
        #PrintState(state)
        
        return np.array(state, dtype=int)
    
    def PlayMove(self, modelOutput):
        modelOutput = modelOutput.sort(reversed=True)
        print(modelOutput)
        
        moveResult = 0
        iters = 0
        while iters < 10:
            move = ConstantData.MODEL_OUTPUT_MOVES[modelOutput[iters]]
            if len(self.engine.board.move_stack) % 2 != 0: FlipMoveOnBoard(move) # If not white's turn
            print(len(self.engine.board.move_stack))
            print(move)
            moveResult = self.engine.PlayMove(move)
            if (moveResult >= 0): break
            iters += 1
            if iters >= 50: 
                print(f"No valid moves found.\n{modelOutput}")
                self.engine.SaveGame()
                return
        return self.RewardFunction(), moveResult, iters
            
    def RewardFunction(self):
        pass
    
    def GetTotalBoardReward(self, isWhite):
        # Loop through board
        # Cross reference with reward list per piece
        
        # Remove spaces
        boardString = str(self.engine.board).replace(" ", "").replace("\n", "")
        if isWhite: boardString.replace("p", ".").replace("n", ".").replace("b", ".").replace("r", ".").replace("q", ".").replace("k", ".")
        else: boardString.replace("P", ".").replace("N", ".").replace("B", ".").replace("R", ".").replace("Q", ".").replace("K", ".")
        iters = 0
        reward = 0
        for x in str(boardString):
            if x == ".":
                iters += 1
                continue
            
            letter = x.lower()
            
            print(self.engine.board.move_stack)
            # Transition between start rewards and end rewards, (initReward / transitionMoves * (transitionMoves - moves)) + (endReward / transitionMoves * moves) = reward
            if letter == "p": reward += (ConstantData.PAWN_START_REWARD[iters] / self.transitionMoves * (self.transitionMoves - (len(self.engine.board.move_stack) / 2))) + (ConstantData.PAWN_END_REWARD[iters] / self.transitionMoves * (len(self.engine.board.move_stack) / 2))
            if letter == "n": reward += ConstantData.KNIGHT_REWARD[iters]
            if letter == "b": reward += ConstantData.BISHOP_REWARD[iters]
            if letter == "r": reward += ConstantData.ROOK_REWARD[iters]
            if letter == "q": reward += ConstantData.QUEEN_REWARD[iters]
            if letter == "k": reward += (ConstantData.KING_START_REWARD[iters] / self.transitionMoves * (self.transitionMoves - (len(self.engine.board.move_stack) / 2))) + (ConstantData.KING_START_REWARD[iters] / self.transitionMoves * (len(self.engine.board.move_stack) / 2))
            
            iters += 1
            
        return reward
    
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done)) # Adds tuple to memory
    
    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE: miniSample = random.sample(self.memory, BATCH_SIZE) # Returns list of tuples
        else: miniSample = self.memory
        
        states, actions, rewards, nextStates, dones = zip(*miniSample)
        
        self.trainer.trainStep(states, actions, rewards, nextStates, dones)
    
    def trainShortMemory(self, state, action, reward, nextState, done):
        self.trainer.trainStep(state, action, reward, nextState, done)
    
    
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

def PrintPieceRewards():
    iters = 9
    transitionMoves = 30
    move_stack = 40
    
    print((ConstantData.KING_START_REWARD[iters] / transitionMoves * (transitionMoves - (move_stack / 2))) + (ConstantData.KING_END_REWARD[iters] / transitionMoves * (move_stack) / 2))
    
    print('\nPawn start Rewards:')
    for x in range(0, int(len(ConstantData.PAWN_START_REWARD))):
        x *= 8
        print(f'\n{ConstantData.PAWN_START_REWARD[x]} {ConstantData.PAWN_START_REWARD[x + 1]} {ConstantData.PAWN_START_REWARD[x + 2]} {ConstantData.PAWN_START_REWARD[x + 3]} {ConstantData.PAWN_START_REWARD[x + 4]} {ConstantData.PAWN_START_REWARD[x + 5]} {ConstantData.PAWN_START_REWARD[x + 6]} {ConstantData.PAWN_START_REWARD[x + 7]}')
        if x >= 55: break
        
    print('\nPawn end Rewards:')
    for x in range(0, int(len(ConstantData.PAWN_END_REWARD))):
        x *= 8
        print(f'\n{ConstantData.PAWN_END_REWARD[x]} {ConstantData.PAWN_END_REWARD[x + 1]} {ConstantData.PAWN_END_REWARD[x + 2]} {ConstantData.PAWN_END_REWARD[x + 3]} {ConstantData.PAWN_END_REWARD[x + 4]} {ConstantData.PAWN_END_REWARD[x + 5]} {ConstantData.PAWN_END_REWARD[x + 6]} {ConstantData.PAWN_END_REWARD[x + 7]}')
        if x >= 55: break
        
    print('\n\nKnight Rewards:')
    for x in range(0, int(len(ConstantData.KNIGHT_REWARD))):
        x *= 8
        print(f'\n{ConstantData.KNIGHT_REWARD[x]} {ConstantData.KNIGHT_REWARD[x + 1]} {ConstantData.KNIGHT_REWARD[x + 2]} {ConstantData.KNIGHT_REWARD[x + 3]} {ConstantData.KNIGHT_REWARD[x + 4]} {ConstantData.KNIGHT_REWARD[x + 5]} {ConstantData.KNIGHT_REWARD[x + 6]} {ConstantData.KNIGHT_REWARD[x + 7]}')
        if x >= 55: break
        
    print('\n\nBishop Rewards:')
    for x in range(0, int(len(ConstantData.BISHOP_REWARD))):
        x *= 8
        print(f'\n{ConstantData.BISHOP_REWARD[x]} {ConstantData.BISHOP_REWARD[x + 1]} {ConstantData.BISHOP_REWARD[x + 2]} {ConstantData.BISHOP_REWARD[x + 3]} {ConstantData.BISHOP_REWARD[x + 4]} {ConstantData.BISHOP_REWARD[x + 5]} {ConstantData.BISHOP_REWARD[x + 6]} {ConstantData.BISHOP_REWARD[x + 7]}')
        if x >= 55: break
        
    print('\n\nRook Rewards:')
    for x in range(0, int(len(ConstantData.ROOK_REWARD))):
        x *= 8
        print(f'\n{ConstantData.ROOK_REWARD[x]} {ConstantData.ROOK_REWARD[x + 1]} {ConstantData.ROOK_REWARD[x + 2]} {ConstantData.ROOK_REWARD[x + 3]} {ConstantData.ROOK_REWARD[x + 4]} {ConstantData.ROOK_REWARD[x + 5]} {ConstantData.ROOK_REWARD[x + 6]} {ConstantData.ROOK_REWARD[x + 7]}')
        if x >= 55: break
        
    print('\n\nQueen Rewards:')
    for x in range(0, int(len(ConstantData.QUEEN_REWARD))):
        x *= 8
        print(f'\n{ConstantData.QUEEN_REWARD[x]} {ConstantData.QUEEN_REWARD[x + 1]} {ConstantData.QUEEN_REWARD[x + 2]} {ConstantData.QUEEN_REWARD[x + 3]} {ConstantData.QUEEN_REWARD[x + 4]} {ConstantData.QUEEN_REWARD[x + 5]} {ConstantData.QUEEN_REWARD[x + 6]} {ConstantData.QUEEN_REWARD[x + 7]}')
        if x >= 55: break
        
    print('\n\nKing start Rewards:')
    for x in range(0, int(len(ConstantData.KING_START_REWARD))):
        x *= 8
        print(f'\n{ConstantData.KING_START_REWARD[x]} {ConstantData.KING_START_REWARD[x + 1]} {ConstantData.KING_START_REWARD[x + 2]} {ConstantData.KING_START_REWARD[x + 3]} {ConstantData.KING_START_REWARD[x + 4]} {ConstantData.KING_START_REWARD[x + 5]} {ConstantData.KING_START_REWARD[x + 6]} {ConstantData.KING_START_REWARD[x + 7]}')
        if x >= 55: break
        
    print('\n\nKing end Rewards:')
    for x in range(0, int(len(ConstantData.KING_END_REWARD))):
        x *= 8
        print(f'\n{ConstantData.KING_END_REWARD[x]} {ConstantData.KING_END_REWARD[x + 1]} {ConstantData.KING_END_REWARD[x + 2]} {ConstantData.KING_END_REWARD[x + 3]} {ConstantData.KING_END_REWARD[x + 4]} {ConstantData.KING_END_REWARD[x + 5]} {ConstantData.KING_END_REWARD[x + 6]} {ConstantData.KING_END_REWARD[x + 7]}')
        if x >= 55: break
        

def Train():
    pass

PrintPieceRewards()

if __name__ == '__main__':
    agent = Agent()
    agent.Loop()