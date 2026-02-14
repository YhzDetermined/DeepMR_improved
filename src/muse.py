from Utils.utils import MutantNameParser


class MUSE:
    def __init__(self, mutation_testing_results, passing_tests_total_count, failing_tests_total_count):
        self.__mutation_testing_results__ = mutation_testing_results
        self.__tot_p = passing_tests_total_count
        self.__tot_f = failing_tests_total_count
        self.__scores = None
        self.__mutants_sus_scores = {}

    def calculate_scores(self):
        self.__scores = dict()
        for (layer_id, layer_mutation_exec_info) in self.__mutation_testing_results__.items():
            term1 = 0.
            if self.__tot_f > 0:
                term1 = float(layer_mutation_exec_info.get_f2p_total_count()) / float(self.__tot_f)
            term2 = 0.
            if self.__tot_p > 0:
                term2 = float(layer_mutation_exec_info.get_p2f_total_count()) / float(self.__tot_p)
            alpha = term1 * term2
            summation = 0.
            for (n_f2p, n_p2f, _, _) in layer_mutation_exec_info.get_mutation_exec_results():
                term1 = 0.
                if self.__tot_f > 0:
                    term1 = float(n_f2p) / float(self.__tot_f)
                term2 = 0.
                if self.__tot_p > 0:
                    term2 = float(n_p2f) / float(self.__tot_p)
                summation = summation + abs(term1 - alpha * term2)
            # for mutant_name,result in layer_mutation_exec_info.get_run_dict().items():
            #     mutant_Tuple=MutantNameParser(mutant_name)
            #     n_f2p,n_p2f=result[0],result[1]
            #     term1 = 0.
            #     if self.__tot_f > 0:
            #         term1 = float(n_f2p) / float(self.__tot_f)
            #     term2 = 0.
            #     if self.__tot_p > 0:
            #         term2 = float(n_p2f) / float(self.__tot_p)
            #     summation = summation + abs(term1 - alpha * term2)
            #     if mutant_Tuple[0]!=-1:
            #         self.__mutants_sus_scores.setdefault(mutant_Tuple,[]).append(abs(term1 - alpha * term2))
            n = layer_mutation_exec_info.get_generated_mutants_total_count()
            if n > 0:
                self.__scores[layer_id] = summation / float(n)
            else:
                self.__scores[layer_id] = 0.

    def get_scores(self):
        return self.__scores

    def get_mutant_sus_score(self):
        return self.__mutants_sus_scores

    def set_mutant_sus_score(self, dict):
        self.__mutants_sus_scores = dict

