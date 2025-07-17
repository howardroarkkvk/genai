import json
from pathlib import Path
from langchain_community.document_loaders import JSONLoader

def metadata_func(record:dict, metadata:dict)->dict:
    metadata['id']=record.get('id')
    metadata['lab']=record.get('label')
    return metadata

file_path=r'D:\DataFiles\kbn'
for file in Path(file_path).glob('*.json'):
    print(file)
    loader=JSONLoader(file_path=file,jq_schema='.menu.items[]',content_key='[.id,.label]|@csv',is_content_key_jq_parsable=True,metadata_func=metadata_func)# jq_schema='.menu.items[].id' --> working model
    document=loader.load()    
    for doc in document:
        print(doc)

    # try:
    #     data=json.loads(Path(file).read_text())
    #     print(data)
    # except Exception as e:
    #     print(e)