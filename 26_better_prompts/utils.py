import sqlite3
import os

def get_schema_info(db_path:str):
    with sqlite3.connect(db_path) as conn:
        cursor=conn.cursor()
        tables=cursor.execute(f"select name from sqlite_master where type ='table'").fetchall()
        schema_info=[]
        for table in tables:
            # print(table[0])
            columns=cursor.execute(f'PRAGMA table_info({table[0]})').fetchall()
            table_info=f'Table Name : {table[0]} \n'
            for col in columns:
                table_info+=f' - {col[1],col[2]}\n'
            schema_info.append(table_info)
        return schema_info




if __name__=='__main__':
    db_path=r'D:\DataFiles\text_to_sql\db\chinook.sqlite'
    print(get_schema_info(db_path=db_path))



