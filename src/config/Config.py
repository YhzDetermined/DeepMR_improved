ExeStatsRows = [
    "model_name",
    "caseCount",
    "totalMutantNum",
    "mutantGenerationTime",
    "selectRatio",
    "selectedMutantNum",
    "executeTime",
    "caseNotExecuteNum",
    "mutantNotFullyExecuteNum",
]
# 是否将结果记录到csv文件中
# 跑实验时请将下面的配置设置为1
isRecord = 1
#记录执行策略下测试执行结果的文件
StatsFileName = "ExeResult.csv"
#存放变异体的文件位置
# mut_file_path = './workdir.tar.gz'
# 跑实验时请改成下面的配置
mut_file_path = 'E://workdir.tar.gz'
#错误定位结束后是否删除变异体
deleteMutFile = True
#错误定位结束后，是否将各层怀疑度写入csv文件中
recordSus = True