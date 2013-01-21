import numpy

points = numpy.random.randn(50,3)
points = points*2+1
points[:,2] = 0

f = open('class1','w')

for i in points:
    for j in i:
        f.write(str(j))
        f.write('\n')
