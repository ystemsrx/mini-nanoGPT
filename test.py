import sqlite3
import pandas as pd
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

def analyze_sqlite_db(db_path):
    """分析SQLite数据库文件，获取基本信息和统计数据"""
    
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件 '{db_path}' 不存在.")
        return
    
    try:
        # 连接到数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 数据库基本信息
        print(f"\n==== 数据库文件: {db_path} ====")
        print(f"文件大小: {os.path.getsize(db_path) / 1024:.2f} KB")
        
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("数据库中没有表.")
            conn.close()
            return
        
        print(f"\n发现 {len(tables)} 个表:")
        for i, table in enumerate(tables, 1):
            table_name = table[0]
            print(f"{i}. {table_name}")
            
            # 表结构信息
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # 创建表结构数据框
            columns_df = pd.DataFrame(columns, columns=['cid', 'name', 'type', 'notnull', 'default_value', 'pk'])
            print("\n表结构:")
            print(tabulate(columns_df[['name', 'type', 'notnull', 'pk']], 
                          headers=['列名', '数据类型', '非空', '主键'], 
                          tablefmt='grid'))
            
            # 记录数量
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"\n记录数量: {count}")
            
            # 显示示例数据 (最多10行)
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
                sample_data = cursor.fetchall()
                
                # 获取列名
                column_names = [desc[0] for desc in cursor.description]
                
                # 转换为DataFrame显示
                sample_df = pd.DataFrame(sample_data, columns=column_names)
                print("\n示例数据 (最多10行):")
                print(tabulate(sample_df, headers='keys', tablefmt='grid'))
                
                # 简单数据可视化 (针对数值型列)
                numeric_columns = sample_df.select_dtypes(include=['int64', 'float64']).columns
                
                if len(numeric_columns) > 0 and count > 1:
                    print("\n数值列统计:")
                    for col in numeric_columns:
                        cursor.execute(f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table_name}")
                        min_val, max_val, avg_val = cursor.fetchone()
                        print(f"  * {col}: 最小值={min_val}, 最大值={max_val}, 平均值={avg_val:.2f}")
            
            print("\n" + "="*50)
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"SQLite错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 设置数据库文件路径
    db_file = "model_registry.db"
    
    # 分析数据库
    analyze_sqlite_db(db_file)