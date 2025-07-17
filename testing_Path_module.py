from pathlib import Path

path='.'
p=Path(path)
list_of_files=[x for x in p.iterdir() if p.is_dir()]
print(list_of_files)

y=list(p.glob('**/*.py'))
print(y)