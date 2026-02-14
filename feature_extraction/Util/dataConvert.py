import numpy as np
def process_and_convert_activations(activations, dic):
    """
    根据指定的层编号提取 activations，并转换为三维张量。

    参数：
    activations: list，模型所有层的激活值。
    dic: dict，部分层编号和对应激活函数类型，格式为 {layer_index: activation_type}。

    返回：
    activations_3d: np.ndarray，三维张量，形状为 (num_selected_layers, num_neurons, batch_size)。
    """
    selected_activations = []

    for layer_index, activation_type in dic.items():
        if layer_index < len(activations):
            # 提取指定层的激活值
            layer_output = activations[layer_index]

            # 将该层的输出展平为 (batch_size, num_neurons)
            batch_size = layer_output.shape[0]
            flattened_output = layer_output.reshape(batch_size, -1)

            # 转置为 (num_neurons, batch_size)
            transposed_output = flattened_output.T

            # 保存结果
            selected_activations.append(transposed_output)
    # 将选定层的结果堆叠为 (num_selected_layers, num_neurons, batch_size)
    activations_3d = np.array(selected_activations, dtype=object)

    return activations_3d

# def convert_activations(activations):
#     new_activations = []
#     for layer_output in activations:
#         num_samples = layer_output.shape[0]
#         # 将每个样本的输出展平成一维向量
#         reshaped_output = layer_output.reshape(num_samples, -1)
#         # 转置以便按神经元索引访问
#         neuron_activations = reshaped_output.T  # 形状为 (神经元数量, 样本数量)
#         new_activations.append(neuron_activations)
#     return new_activations
def convert_activations(activations):
    if isinstance(activations, list):
        new_activations = []
        for layer_output in activations:
            # layer_output 的形状为 (样本数量, 神经元数量)
            num_samples = layer_output.shape[0]
            # 如果输出是一维的，需要在 axis=1 增加一个维度
            if len(layer_output.shape) == 1:
                layer_output = layer_output.reshape(num_samples, 1)
            # 转置以便按神经元索引访问
            neuron_activations = layer_output.T  # 形状为 (神经元数量, 样本数量)
            new_activations.append(neuron_activations)
    else:
        # 如果 activations 是一个二维数组
        num_samples = activations.shape[0]
        # 如果输出是一维的，需要在 axis=1 增加一个维度
        if len(activations.shape) == 1:
            activations = activations.reshape(num_samples, 1)
        # 转置以便按神经元索引访问
        neuron_activations = activations.T  # 形状为 (神经元数量, 样本数量)
        new_activations = [neuron_activations]
    return new_activations