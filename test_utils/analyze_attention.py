import numpy as np

def analyze_attention_matrix(model_type):
    """分析已保存的注意力矩阵"""
    # 加载保存的注意力矩阵
    file_path = f'scores_layer4_before_softmax.npy'
    scores = np.load(file_path)
    
    # 打印矩阵的基本信息
    print(f"\n{model_type}模型第4层注意力矩阵信息:")
    print(f"完整矩阵形状: {scores.shape}")  # 应该是 (1, num_heads, seq_len, seq_len)
    
    # 获取第一个注意力头的矩阵
    head_matrix = scores[0, 0]  # 去掉batch维度和head维度
    print(f"第一个注意力头矩阵形状: {head_matrix.shape}")
    
    # 保存head_matrix
    head_matrix_path = f'1_head_matrix.npy'
    np.save(head_matrix_path, head_matrix)
    print(f"已保存第一个注意力头矩阵到: {head_matrix_path}")
    
    # 检查是否为下三角矩阵
    upper_tri = np.triu(head_matrix, k=1)  # 获取上三角部分（不包括对角线）
    lower_tri = np.tril(head_matrix)  # 获取下三角部分（包括对角线）
    mask_value = -3.4028235e+38  # 使用具体的掩码值
    
    # 检查上三角部分是否都是掩码值
    is_upper_masked = np.allclose(upper_tri, mask_value, rtol=1e-5)
    
    # 检查下三角部分是否都不等于掩码值
    is_lower_valid = not np.any(np.isclose(lower_tri, mask_value, rtol=1e-5))
    
    print(f"\n下三角矩阵检查:")
    print(f"1. 上三角部分是否都是掩码值: {is_upper_masked}")
    print(f"2. 下三角部分是否都不等于掩码值: {is_lower_valid}")
    print(f"3. 上三角部分最大值: {np.max(upper_tri)}")
    print(f"4. 下三角部分最大值: {np.max(lower_tri)}")
    print(f"5. 使用的掩码值: {mask_value}")
    
    # 打印矩阵的具体值
    print("\n完整的注意力矩阵值:")
    print(head_matrix)

if __name__ == "__main__":
    # 分析所有模型类型的注意力矩阵
    model_types = ["eval"]
    
    for model_type in model_types:
        try:
            analyze_attention_matrix(model_type)
        except FileNotFoundError:
            print(f"\n未找到{model_type}模型的注意力矩阵文件") 