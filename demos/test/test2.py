import numpy as np
import os
import operator

a = np.array([[3,'/root/ylong/workspace/openface/demos/web/captured/aligned/longyg/124.png'], [3, '/root/ylong/workspace/openface/demos/web/captured/aligned/longyg/169.png']])

b = a[:, 1]
print(b)

list1 = map(os.path.dirname, b)
print(list1)

list2 = map(os.path.split, list1)
print(list2)

list3 = map(operator.itemgetter(1), list2)
print(list3)
