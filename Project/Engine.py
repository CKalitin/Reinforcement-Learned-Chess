import chess

class Engine:
    def __init__(self):
        self.board = chess.Board()
        pass
    
    def BeginGame(self):
        self.board = chess.Board()
        print(self.board)
        pass
    
    def EndGame(self):
        pass
    
    def PlayMove(self, move):
        try:
            # Move example "e2e4"
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
    
    def SaveGame(self):
        pass