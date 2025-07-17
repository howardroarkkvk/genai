from langchain_text_splitters import RecursiveCharacterTextSplitter

text='hi my name is vamshi i m an engineer by professoin , i m interested in programming, testing and learning new' \
'software technologies'
# text = """What I Worked On

# February 2021

# Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. 
# I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. 
# They had hardly any plot, just characters with strong feelings, which I imagined made them deep.

# The first programs I tried writing were on the IBM 1401 that our school district used for what was then called 
# "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the 
# basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini 
# Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — 
# sitting up on a raised floor under bright fluorescent lights.
# """

text_splitter=RecursiveCharacterTextSplitter(chunk_size=20,chunk_overlap=10,length_function=len,separators=['\n\n','.',''])
chunks=text_splitter.split_text(text)
print('Total chunks with overlap ', len(chunks))
for chunk in chunks:
    print(len(chunk),chunk)
print()


