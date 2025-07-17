import sqlite3
from config_reader import settings



def get_chunks_by_field(db_file_path:str):
    with sqlite3.connect(database=db_file_path) as conn:
        cursor=conn.cursor()
        result=cursor.execute(f"select name from sqlite_master where type='table'").fetchall()
        chunks=[]
        for (table,) in result:
            cols=cursor.execute(f'PRAGMA table_info({table})').fetchall()
            # print(cols)

            for col in cols:
                chunk={'id':col[0],
                       "text":f"Table:{table} , column:{col[1]}, Type: {col[2]} ",
                       'metadata':{'table':table , 'column':col[1], 'Type': col[2]}
                       }
                chunks.append(chunk)

        
        return chunks
    

def get_chunks_by_table(db_file_path:str):
    with sqlite3.connect(database=db_file_path) as conn:
        cursor=conn.cursor()
        result=cursor.execute(f"select name from sqlite_master where type='table'").fetchall()
        chunks=[]
        seq=1
        for (table,) in result:
            text=[]
            text.append(f'Table: {table}')
            cols=cursor.execute(f'PRAGMA table_info({table})')
            for col in cols:
                text.append(f' -({col[1]},{col[2]})')
            text='\n'.join(text)
            chunk={'id':seq,'text':text,'metadata':{'table':table}}
            chunks.append(chunk)
            seq=seq+1
        return chunks
        
            
if __name__=='__main__':
    db_file_path='D:/DataFiles/db_explorer/db/chinook.sqlite'
    print(get_chunks_by_field(db_file_path=db_file_path))
    print(get_chunks_by_table(db_file_path=db_file_path))