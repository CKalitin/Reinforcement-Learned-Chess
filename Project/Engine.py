import chess
import datetime
import os
from chess import pgn
import traceback
import Agent

almostPromotionFEN = "8/1P2Q3/8/k7/8/8/6K1/8"

class Engine:
    def __init__(self):
        pass
    
    def BeginGame(self):
        now = datetime.datetime.now()
        self.board = chess.Board()#almostPromotionFEN)
        self.gameName = f'Neural-Net-v1_{now.strftime("%m:%d:%H:%M:%S").replace(":", ".")}'
    
    def EndGame(self):
        self.SaveGame()
    
    def PlayMove(self, move):
        try:
            # Move example "e2e4"
            if (self.CheckMoveLegal(move)) == 0:
                #print(f"Move Illegal: {move}")
                return -1
            
            self.board.push_uci(move)
            
            if self.board.outcome() != None:
                return self.board.outcome().termination.value
            # https://python-chess.readthedocs.io/en/latest/core.html#outcome
            
            return 0
        except Exception as error:
            print(f"Invalid move: {move} | {error.with_traceback()}")
            return -1
            
    # Must be done before game reset
    def SaveGame(self):
        now = datetime.datetime.now()
        path = os.path.join(os.getcwd(), f'Project\games\Game-{self.gameName}.txt')
        
        game = pgn.Game()
        game.headers["Event"] = self.gameName
        game.headers["Site"] = "192.168.1.73"
        game.headers["Date"] = now.strftime("%x")
        game.headers["Round"] = "x"
        game.headers["White"] = "AI"
        game.headers["Black"] = "Another AI?"
        if (self.board.outcome() != None): game.headers["Result"] = str(self.board.outcome())
        
        node = game.add_variation(self.board.move_stack[0])
        for move in self.board.move_stack:
            if move == self.board.move_stack[0]: continue
            node = node.add_variation(move)
        node.comment = ""
        
        f = None
        if (os.path.isfile(path)): f = open(path, "w")
        else: f = open(path, "x")
        
        try:
            f.write(str(game))
        except Exception as error:
            f.write(f"\nError writing game to file:\n{error}\n\n")
            f.write(str(self.board.move_stack))
        f.close()
    
    # Return 1 if legal, return 0 if illegal
    def CheckMoveLegal(self, move):
        legalMoves = list(self.board.legal_moves)
        for x in legalMoves:
            if (str(move) == str(x)): return 1
        return 0
        