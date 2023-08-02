import Engine as E

class Agent:
    def __init__(self):
        self.engine = E.Engine()
        
    def Loop(self):
        self.engine.BeginGame()
        while(True):
            move=input("Move:")
            self.engine.PlayMove(move)
            self.engine.SaveGame()

if __name__ == '__main__':
    agent = Agent()
    agent.Loop()