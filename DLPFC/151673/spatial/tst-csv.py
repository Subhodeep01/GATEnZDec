import pandas as pd

# 读取 .txt 文件，假设是以制表符分隔的文件
txt_file_path = './tissue_positions_list.txt'
csv_file_path = './tissue_positions_list.csv'

# 读取txt文件
df = pd.read_table(txt_file_path, sep=',')

# 将其保存为csv文件
df.to_csv(csv_file_path, index=False)

print(f"文件已成功转换为 {csv_file_path}")
