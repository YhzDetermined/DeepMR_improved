class CurExeInfo:
    def __init__(self):
        self.layer_id = ""
        self.proportion = 0
        self.sus = 0.0
        self.exeTime = 0
        self.caseNum = 0
        self.mutant_count = 0

    def resetLayer(self,layer_id):
        self.sus = 0.0
        self.exeTime = 0
        self.caseNum = 0
        self.mutant_count = 0
        self.layer_id = layer_id
        self.proportion = 0

    def resetProportion(self,proportion):
        self.proportion = proportion

    def updateSus(self,val):
        self.sus = max(self.sus,val)
    def addExe(self,t:float):
        self.exeTime += t

    def addCase(self,num:int):
        self.caseNum += num

    def MutCountIncrAdd(self):
        self.mutant_count += 1