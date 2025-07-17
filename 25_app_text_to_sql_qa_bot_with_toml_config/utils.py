import sqlite3
from config_reader import settings
import os


def get_chunks_by_field(db_file_path:str):
    with sqlite3.connect(db_file_path) as conn:
        cursor=conn.cursor()
        tables=cursor.execute("select name from sqlite_master where type ='table'").fetchall()
        seq=1
        chunks=[]
        for (table,) in tables:
            cols=cursor.execute(f"PRAGMA table_info({table})").fetchall()
            for col in cols:
                chunks.append({'id':seq,'text':f" Table:{table} column_name:{col[1]} column_type:{col[2]}","metadata":{'table':table,'column_name':col[1],'column_type':col[2]}})
                seq+=1

        return chunks
    
def get_chunks_by_table(db_file_path:str):

    with sqlite3.connect(db_file_path) as conn:
        cursor=conn.cursor()
        tables=cursor.execute("select name from sqlite_master where type='table'").fetchall()

        seq=1
        chunks=[]

        for (table,) in tables:
            text=[]
            text.append(f'Table: {table}\n')
            cols=cursor.execute(f'PRAGMA table_info({table})').fetchall()
            for col in cols:
                text.append(f" - {col[1],col[2]}")
            text='\n'.join(text)

            chunk={'id':seq,'text':text,'metadata':{'table':table}}
            chunks.append(chunk)
            seq+=1

        return chunks
    
def execute_query(db_file_path:str,sql:str)-> str:
    with sqlite3.connect(db_file_path) as conn:
        cursor=conn.cursor()
        try:
            results=cursor.execute(sql).fetchall()
            final=results[0][0] if results else 0
            return int(final)
        except Exception :
            return -1


if __name__=='__main__':
    db_file_path=os.path.join(settings.file_paths.src_dir,settings.file_paths.db_file)
    # chunks=get_chunks_by_field(db_file_path)
    # print(chunks)
    chunks=get_chunks_by_table(db_file_path)
    print(chunks)






            
