class ExecutionStatistic:
    def __init__(self):
        self.selected_ratio = 0 #选择的变异体比例
        self.generateTime = 0 #变异体生成时间
        self.executeTime = 0 # 测试用例执行时间
        self.caseNotExecuteNum = 0  #减少执行的测试用例数
        self.mutantNotExecuteAllNum = 0 #变异体执行策略减少执行的变异体数
        self.caseCount = 0 #测试用例数
        self.totalMutantNum = 0 #变异体数
        self.selectedMutantNum = 0 #选择的变异体数目

    def setExecuteTime(self,time):
        self.executeTime = time

    def setGenerateTime(self,time):
        self.generateTime = time

    def setSelectRatio(self,ratio):
        self.selected_ratio = ratio






