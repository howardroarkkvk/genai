from bert_score import score
from rouge import Rouge


reference='it is not cold'
response='it is freezing'

p,r,f=score([response],[reference],lang='en')
print(p.item())
print(r.item())
print(f.item())

scores=Rouge().get_scores(response,reference)
rouge_1=scores[0]['rouge-1']
print(rouge_1['p'])
print(rouge_1['r'])
print(rouge_1['f'])

rouge_2=scores[0]['rouge-2']
print(rouge_2['p'])
print(rouge_2['r'])
print(rouge_2['f'])