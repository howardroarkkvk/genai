#how to use lambda function in python

# add=lambda x:str(x)[0]
# print(add(100))

# dict1={'a':3,'b':2,'c':1}
# for k,v in dict1.items():
#     print(k,v)

# for x in dict1.items():
#     print(x[1])
# y=dict(sorted(dict1.items(),key=lambda x:x[1],reverse=False))
# print(y)

list1=[5,6,4,3,2,1]
# sorted_list=sorted(list1,key=lambda x:x,reverse=True)
# print(sorted_list)

# square= lambda x,y:x**y
# print(square(5,2))

for i in range(len(list1)):
    for j in range(i+1,len(list1)):
        if list1[i]<list1[j]:
            list1[j],list1[i]=list1[i],list1[j]
print(list1)