import chess
import datetime
import os
from chess import pgn

class Engine:
    def __init__(self):
        self.board = chess.Board()
        pass
    
    def BeginGame(self):
        now = datetime.datetime.now()
        self.board = chess.Board()
        self.gameName = f'Neural-Net-v1_{now.strftime("%H:%M:%S").replace(":", ".")}'
        print(self.board)
        pass
    
    def EndGame(self):
        pass
    
    def PlayMove(self, move):
        try:
            # Move example "e2e4"
            if (self.CheckMoveLegal(move)) == 0:
                print(f"Move Illegal: {move}")
                return -1
            
            pushMove = chess.Move.from_uci(move)
            self.board.push(pushMove)
            
            print(self.board)
            
            if self.board.outcome() != None:
                print(self.board.outcome())
                return self.board.outcome()
            # https://python-chess.readthedocs.io/en/latest/core.html#outcome
            
            return 0
        except Exception as error:
            print(f"Invalid move: {move} | {error}")
            return -1
            
    
    def GetBoardState(self):
        pass
    
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
        #game.headers["Result"] = self.board.outcome()
        
        node = game.add_variation(self.board.move_stack[0])
        for move in self.board.move_stack:
            if move == self.board.move_stack[0]: continue
            node = node.add_variation(move)
        node.comment = ""
        
        f = None
        if (os.path.isfile(path)): f = open(path, "w")
        else: f = open(path, "x")
        
        f.write(str(game))
        f.close()
    
    # Return 1 if legal, return 0 if illegal
    def CheckMoveLegal(self, move):
        legalMoves = list(self.board.legal_moves)
        for x in legalMoves:
            if (str(move) == str(x)): return 1
        return 0
        