import Engine as E
import numpy as np

class Agent:
    def __init__(self):
        self.engine = E.Engine()
        
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
    
    
def PrintState(state):
    output = ""
    for x in range(0, int(len(state) / 8)):
        if x % 8 == 0: output += "\n"
        x *= 8
        output += f'\n{state[x]} {state[x + 1]} {state[x + 2]} {state[x + 3]} {state[x + 4]} {state[x + 5]} {state[x + 6]} {state[x + 7]}'
    output += "\n"
    print(output)

def ModelOutputToMove():
    # because it might be playing as black
    pass

if __name__ == '__main__':
    agent = Agent()
    agent.Loop()