from __future__ import print_function
from util import *

d1 = get_dataset_1()
d2 = get_dataset_2()

f1 = open("dataset1", 'w')
f2 = open("dataset2", 'w')

for (x, y), l in zip(*d1):
    print(x, y, l, file=f1)

for (x, y), l in zip(*d2):
    print(x, y, l, file=f2)
