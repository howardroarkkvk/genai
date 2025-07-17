from bert_score import score
from rouge import Rouge


reference='On the eve of polling day, UKIP leader Nigel Farage told interviewers he would be "for the chop" if he failed to get elected in the Kent seat of South Thanet.'
response='Nigel Farage, despite losing his seat in the 2015 election, remains a controversial figure leading UKIP, a party that gained popularity through his anti-EU and anti-immigration stance. '

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