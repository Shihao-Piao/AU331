import math

mp = 0.027
lp = 0.191
ip = 0.153
mp1 = 0.008
mp2 = 0.019
lp1 = 0.171
lp2 = 0.19
jp = 0
r = 0.0826
g = 9.8
jm = 0.00003
marm = 0.08
jeq = 0.000184
rm = 8.7
kt = 0.0333
km = 0.02797

f = 2.353
jp = (mp*lp*g)/(4*math.pi*math.pi*f*f)

A = [
    [0,0,1,0],
    [0,0,0,1],
    [0,0,0,0],
    [0,0,0,0]
]

B = [0,0,0,0]
C = [
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
]
D = [0,0,0,0]

A[2][1] = (r*mp*mp*lp*lp*g)/(jp*jeq+mp*lp*lp*jeq+jp*mp*r*r)
A[2][2] = - (kt*km*(jp+mp*lp*lp))/(rm*(jp*jeq+mp*lp*lp*jeq+jp*mp*r*r))
A[3][1] = - (mp*lp*g*(jeq+mp*r*r))/(jp*jeq+mp*lp*lp*jeq+jp*mp*r*r)
A[3][2] = (mp*lp*kt*r*km)/(rm*(jp*jeq+mp*lp*lp*jeq+jp*mp*r*r))
B[2] = (kt*(jp+mp*lp*lp))/(rm*(jp*jeq+mp*lp*lp*jeq+jp*mp*r*r))
B[3] = (mp*lp*kt*r)/(rm*(jp*jeq+mp*lp*lp*jeq+jp*mp*r*r))

print("jp = ",jp)
print('A:')
for i in range(4):
    print(A[i])
print('\n')
print('B:',B)
print('\n')
print('C:')
for i in range(4):
    print(C[i])
print('\n')
print('D:',D)