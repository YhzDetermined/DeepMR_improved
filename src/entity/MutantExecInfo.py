class MutantExecInfo:
    def __init__(self,mutant_name):
        self.__mutant_name = mutant_name
        self.upper_sus = 0.0
        self.real_sus = 0.0
        self.a_k_f = 0
        self.a_n_f = 0

    def set_f_info(self,kill,not_kill):
        self.a_k_f = kill
        self.a_n_f = not_kill

    def getName(self):
        return self.__mutant_name
