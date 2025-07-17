

import sqlite3
import os

# using sqllite3 provides api interfact to the sqlite library
# using connect (db_path) ,db_path contains the sqlite database file (it has the file in database file format)
# we establish connection to the database file using sqllite3.connect(file_path)
# once connection is established, we have to create a cursor  to query the database.
# a cursor is an object which is established to make the connection to execute the querie
#cursor.execute( <sql_query >) -> this is used to execute the sql query
# cursor.fetchall()--> retrieves the query results as list....


def get_schema_info(db_path:str)->str:

    with sqlite3.connect(db_path) as conn:
        cursor=conn.cursor()
        cursor.execute("select name from sqlite_master where type ='table'")
        output=cursor.fetchall()
    schema_info=[]
    for (table,) in output:
        table_info=f'Table: {table} \n'
        cursor.execute(f'pragma table_info({table})')
        result=cursor.fetchall()
        table_info+='\n'.join([' - ' + col_info[1]+ ' ' + col_info[2] for col_info in result])
        # print(table_info)
        schema_info.append(table_info)
    final_schema='\n'.join(schema_info)
    # print(final_schema)
    return final_schema

if __name__=='__main__':

    src_path=r'D:\DataFiles\text_to_sql'
    db_path=r'db\chinook.sqlite'
    final_path=os.path.join(src_path,db_path)
    print(get_schema_info(final_path))
    





