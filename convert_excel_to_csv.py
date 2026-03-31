#!/usr/bin/env python3
"""
将 data_new 目录下的 Excel 文件（扩展名为.csv但实际上是xlsx格式）
转换为与参考文件格式相同的 CSV 格式。

参考文件格式：Date,Open,High,Low,Close,Adj Close,Volume
"""

import pandas as pd
import os
import glob
import numpy as np

def find_column(columns, patterns):
    """根据模式列表查找匹配的列名"""
    for pattern in patterns:
        for col in columns:
            if isinstance(col, str) and pattern.lower() in col.lower():
                return col
    return None

def convert_excel_to_csv(input_path, output_path):
    """将 Excel 文件转换为标准 CSV 格式"""
    # 读取 Excel 文件
    df = pd.read_excel(input_path, engine='openpyxl')
    
    # 标准化列名（去除空格）
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    columns = df.columns.tolist()
    
    print(f"  列名: {columns}")
    
    # 定义列名映射模式（中文关键词）
    date_patterns = ['日期', 'date']
    open_patterns = ['开盘价', 'opening price']
    high_patterns = ['最高价', 'highest price']
    low_patterns = ['最低价', 'lowest price']
    close_patterns = ['收盘价', 'closing price']
    volume_patterns = ['成交量', 'volume']
    
    # 查找各列
    date_col = find_column(columns, date_patterns)
    open_col = find_column(columns, open_patterns)
    high_col = find_column(columns, high_patterns)
    low_col = find_column(columns, low_patterns)
    close_col = find_column(columns, close_patterns)
    volume_col = find_column(columns, volume_patterns)
    
    # 检查必要的列是否存在
    missing_cols = []
    if not date_col:
        missing_cols.append('Date')
    if not close_col:
        missing_cols.append('Close')
    
    if missing_cols:
        print(f"  跳过: 缺少必要列 {missing_cols}")
        return None
    
    # 创建新的 DataFrame，使用标准列名
    new_df = pd.DataFrame()
    new_df['Date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    
    # 如果某些列不存在，使用 Close 列填充
    if open_col:
        new_df['Open'] = pd.to_numeric(df[open_col], errors='coerce')
    else:
        new_df['Open'] = pd.to_numeric(df[close_col], errors='coerce')
        
    if high_col:
        new_df['High'] = pd.to_numeric(df[high_col], errors='coerce')
    else:
        new_df['High'] = pd.to_numeric(df[close_col], errors='coerce')
        
    if low_col:
        new_df['Low'] = pd.to_numeric(df[low_col], errors='coerce')
    else:
        new_df['Low'] = pd.to_numeric(df[close_col], errors='coerce')
    
    new_df['Close'] = pd.to_numeric(df[close_col], errors='coerce')
    
    # Adj Close 与 Close 相同（如果没有复权数据）
    new_df['Adj Close'] = new_df['Close']
    
    if volume_col:
        # 处理成交量，允许 NA 值
        vol = pd.to_numeric(df[volume_col], errors='coerce')
        # 将 NA 值填充为 0 并转换为整数
        new_df['Volume'] = vol.fillna(0).astype(int)
    else:
        new_df['Volume'] = 0
    
    # 删除所有价格都为 NA 的行
    new_df = new_df.dropna(subset=['Open', 'High', 'Low', 'Close'], how='all')
    
    # 保存为 CSV
    new_df.to_csv(output_path, index=False)
    print(f"  已转换: {len(new_df)} 行 -> {output_path}")
    return new_df

def main():
    input_dir = "/home/zhoukegu/GNN_stocRL_project/data_new"
    output_dir = "/home/zhoukegu/GNN_stocRL_project/data_output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 .csv 文件（实际上是 Excel 文件）
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    files.sort()
    
    print(f"找到 {len(files)} 个文件需要转换")
    print("=" * 60)
    
    success_count = 0
    failed_files = []
    
    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        # 清理文件名（去除空格，统一格式）
        clean_name = filename.replace(" ", "").replace("__", "")
        # 确保文件名是大写的股票代码
        clean_name = clean_name.upper()
        output_path = os.path.join(output_dir, clean_name)
        
        print(f"\n[{i}/{len(files)}] 处理: {filename}")
        try:
            result = convert_excel_to_csv(file_path, output_path)
            if result is not None:
                success_count += 1
            else:
                failed_files.append(filename)
        except Exception as e:
            print(f"  错误: {e}")
            failed_files.append(filename)
    
    print("\n" + "=" * 60)
    print(f"转换完成! 成功: {success_count}/{len(files)}")
    if failed_files:
        print(f"失败的文件: {failed_files}")

if __name__ == "__main__":
    main()
