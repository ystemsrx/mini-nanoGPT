import sqlite3
import pandas as pd
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

def analyze_sqlite_db(db_path):
    """Analyze SQLite database file, obtain basic information and statistics"""
    
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' does not exist.")
        return
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Database basic information
        print(f"\n==== Database file: {db_path} ====")
        print(f"File size: {os.path.getsize(db_path) / 1024:.2f} KB")
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables in database.")
            conn.close()
            return
        
        print(f"\nFound {len(tables)} tables:")
        for i, table in enumerate(tables, 1):
            table_name = table[0]
            print(f"{i}. {table_name}")
            
            # Table structure information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Create table structure dataframe
            columns_df = pd.DataFrame(columns, columns=['cid', 'name', 'type', 'notnull', 'default_value', 'pk'])
            print("\nTable structure:")
            print(tabulate(columns_df[['name', 'type', 'notnull', 'pk']], 
                          headers=['Column', 'Data Type', 'Not Null', 'PK'], 
                          tablefmt='grid'))
            
            # Record count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"\nRecord count: {count}")
            
            # Display sample data (max 10 rows)
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
                sample_data = cursor.fetchall()
                
                # Get column names
                column_names = [desc[0] for desc in cursor.description]
                
                # Convert to DataFrame for display
                sample_df = pd.DataFrame(sample_data, columns=column_names)
                print("\nSample data (max 10 rows):")
                print(tabulate(sample_df, headers='keys', tablefmt='grid'))
                
                # Simple data visualization (for numeric columns)
                numeric_columns = sample_df.select_dtypes(include=['int64', 'float64']).columns
                
                if len(numeric_columns) > 0 and count > 1:
                    print("\nNumeric column statistics:")
                    for col in numeric_columns:
                        cursor.execute(f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table_name}")
                        min_val, max_val, avg_val = cursor.fetchone()
                        print(f"  * {col}: Min={min_val}, Max={max_val}, Avg={avg_val:.2f}")
            
            print("\n" + "="*50)
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Set database file path
    db_file = "assets/model_registry.db"
    
    # Analyze database
    analyze_sqlite_db(db_file)