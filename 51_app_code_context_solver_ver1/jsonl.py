import json

# reads a jsonl file and converts in to a python list of dictionaries


def read_jsonl(filename):

    with open(filename,'r',encoding='utf-8') as file:
        lines=[]
        for line in file:
            lines.append(json.loads(line))
    return lines

# writer a python list of dictionaries in to jsonl file
def write_jsonl(filename,lines):
    with open(filename,'w',encodding='utf-8') as file:
        for line in lines:
            file.write(json.dumps(line)+'\n')

# loads is reading from json and dumps is writing in to json
# encoding='utf-8' allows all the characters....

