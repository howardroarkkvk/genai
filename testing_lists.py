from itertools import chain
all_list=[[1,2],[3,4,5],[6,7,8]]
final=[list2  for list1 in all_list for list2 in list1]
print(final )

x=list(chain.from_iterable(all_list))
print(x)
