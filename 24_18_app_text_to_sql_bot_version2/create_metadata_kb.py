import json

import os 
import sqlite3

def create_schema_info(src_dir:str,db_dir:str,json_file_name:str):

    
    json_path=os.path.join(src_dir,"kb")
    if not os.path.exists(json_path):
        os.makedirs(json_path)

    json_file_path=os.path.join(json_path,json_file_name)

    db_file=os.path.join(src_dir,db_dir)

    with sqlite3.connect(db_file) as conn:
        cursor=conn.cursor()
        tables=cursor.execute("select name from sqlite_master where type='table'").fetchall()
        list2=[    {             "text": f"Table: {table[0]}, Column: {col[1]}, Type: {col[2]}",
                    "metadata": {"table": table[0], "column": col[1], "type": col[2]},} for table in tables for col in cursor.execute(f"PRAGMA table_info({table[0]})").fetchall()]
        # print(list2)

    with open(json_file_path,'w') as f:
        json.dump(list2,f,indent=2)

if __name__=='__main__':
    src_dir=r'D:\DataFiles\text_to_sql'
    db_dir=r'db\chinook.sqlite'
    json_file_name='chinook_schema.json'
    create_schema_info(src_dir,db_dir,json_file_name)

