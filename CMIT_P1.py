# K Chen @ strath.ac.uk
# CMIT 2024 Python Lecture 1 == CMIT_P1.ipynb  [jupyter nbconvert --to python notebook.ipynb]
''' Content in L1  [To launch juypter, type "python -m jupyter notebook" or "juypter notebook"]
1) Basic commands for scalar numbers and texts
2) Programming basics:  
        (i) "if" coditional, (ii) "for" loop,  (iii) arrays, matrices, tensors
3) HW Exercise 1
'''
A = 2024.6

print('Task 1 --- Basic commands for scalar numbers / print')

a = 2024  # 1) Basic commands for scalar numbers and texts ----------------------------------------
b = 6  # June
c = 2024.2 / 4
d = 2024 % 3  #Compute remainder
print('a b c =', a, b, c,  'd =',d,  '\n')

print('Task 2 --- Basic commands for processing/extracting texts')


F = 'CMIT_P1.ipynb' # This refers to the file that we are using   [STRING]
a = F[0:2];  b=F[1:4]    # Noting for ranges  0 start and  end-1   
What_is = F[-1]
print('Strings')
print('a =',a,  ' --> b =', b, '  Us =',F[0:4], ' ',
       What_is, 'last two =', F[-2:], '\n')

print('Task 3 --- Basic commands for spliting a text to extract extension')
print('F =', F)
a = F.split('.')[0]
b = F.split('.')[1];   
print(a, '(name) and (ext)', b, ' or', F.split('.')[-1],  '\n' )


print( 'This is running: ', F.split('\\')[-1],  '\n' )

print('Task 4 --- Basic commands for calling a math function')

import math

a = math.exp(1.0);  b=math.pi
print(a, b)


# 2) Programming basics:           (i) "if" coditional,
print('\nTask 5 --- Programming basics:           (i) "if" coditional')

import numpy
import numpy as np    # NB for "if"  (a) :   (b) tab 111111111111111111111111111111
x = numpy.random.rand() - 0.5
y = np.random.rand() - 0.5
if x >= 0.0:
    print('non-negative x', x)
else:
    print('\tnegative x', x)

if y >= 0.1:
    print('positive y and y>=0.1', y)
else:
    print('\tless than 0.1', y)


# Task: How to compute the series 
# $s1=\displaystyle\sum_{i=1}^n\sin(0.025 i\pi)$ and 
# $s2=\displaystyle\sum_{j=0}^n \frac{x^j}{j!}$ 
print('\nTask 6 --- Programming basics:           (ii) "for" loop')

print('\tfor -- example 1')
n = 8;   s1 = 0    ## (ii) "for" loop
for i in range(1, n+1):
    s1 = s1 + math.sin(0.025*i*math.pi)   ##  or commonly    s1 += math.sin(0.025*i*math.pi)  
    print(i, 's1 =', s1, ' where n =',n)

print('\n\tfor -- example 2')

n = 3;   s2 = 0;  x=0.12
for j in range(n+1):
    t = x**j / math.factorial(j)
    print(j, t)
    s2 += t
print('When x=', x, ' s2 =', s2, ' Noting exp(x) =', math.exp(x) )

print('\n\tfor -- example 2 (efficient)')

s2 = 1;  x=0.12;  t=x;  n=25   # Efficient code // "if" for less printing
for j in range(1,n+1):
    if j % 5 == 0 or j==1:
        print(j, t)
    s2 += t
    t = x*t/(j+1)       # note when j=1,  t=x^2 / 2!
print('When x=', x, ',  s2 = %.8f' % s2, 'NB exp(x) = %.8f' % math.exp(x), ' er={:.2e}'.format(s2-math.exp(x)) )


# (iii) arrays, matrices, tensors  // HOW TO use "apppend" to build up a LIST. 
print('\nTask 7 --- Programming basics:           (iii) arrays, matrices, tensors')
print('\n\t vectors a b c')
import numpy as np    # most common but there are other ways
a = np.arange(4)
b = np.arange(2,7)    # Moast identical to Matlab's linspace
print('a=',a, 'b=',b)
h = 0.125
c = np.arange(3.0, 4.0+h, h);   print('c =', c)
print('Here Length: a b c =', len(a), len(b), len(c), ' and shape a =', a.shape)


# Task :  generate a vector (array) v with $v_i=i\pi$ and a matrix A with $A_{i,j}=\sin( 0.25i + 0.32j ) +\delta_{i,j}$, where $i,j=0,...,4$ and $\delta_{i,i}=1$
print('\n\t vector v and matrix B   (by append)')

v = [];  B=[]
for i in range( 5 ):  # dim = 4
    x = i*math.pi;    v.append( x )
    B.append([])   # This is needed for notation [[]] for matrix
    for j in range( 5 ):
        y = math.sin( i*0.25 + j*0.32 );  
        if i==j:
            y=y+1
        B[i].append( y )
print('v =',np.array(v).astype(float), '\n')
print('B =', np.array(B))

print('\n\t Norm of matrix B')

# Matrix norm
from numpy import linalg as LA
B = np.array(B);   print('B shape (dim) =',B.shape)
B_n = LA.norm( B, ord='fro')
print('||B||_F = ', B_n)

print('\n\t Solution of a linear system Bx = b  (NB. matrix-vector product)')
b = np.array(v)
x =  LA.solve(B, b)  # Solve Bx = b
r = np.dot(B,x)-b
print('SOL x=',x)
print('Shapes =', B.shape,b.shape,x.shape)
print('Err =', r, ' Norm = %.3e' % LA.norm(r))

print('\nTask 8 --- np.array to torch array (used a lot later in DL)')

# Tensors (Deep Learning use later)
import torch
A = np.zeros( (7,7,3) )
print(A.shape, 'typical color image (after imread and in plotting) numpy\n')

A1 = torch.from_numpy(A)
print(A1.size(), 'typical color image (torch tensor) same shape')

A2 = A1.permute([2,0,1])
print(A2.size(), 'typical color image (torch tensor) new shape towards DL')

A3 = A2[None,:]
print(A3.size(), 'typical color image (torch tensor) new shape ready for DL (with batch)\n')

print('\nTask 9 --- Form a typical torch array (batch,depth,dx,dy)')
A = np.zeros( (1,3,7,7) )
B = np.zeros( (4,3,7,7) )
print(np.shape(A), 'typical format for DL operations:    single batch data (3 channels)')
print(np.shape(B), 'typical format for DL operations:  when batch size = 4 (3 channels)')

A = torch.rand( 2, 3,7,7 )
print(A.shape, 'Torch rand more flexible for tensor shape (np cannot do > 2 dim)\n')

# 
print('\nHome work 1 --- Generate a matrix and compute its condition number\n')

s='TASK:  If not sure about the maths concepts, check wiki or other sources.\n\
   Write a code to generate the Hilbert matrix $H$ with $H_{ij}=1/(i+j-1)$ for $i,j=1,...,n$.\n\
   Then compute its condition number $cond(H)$ for $n=4, 8$ by numpy.linalg.cond\n\
   If your code runs okay,  just show it quickly and your answers on screen in the next lecture\n'
print(s)

import datetime
time0 = datetime.datetime.now().strftime("%Y %m %d @ %H:%M:%S"); print('\t time =', time0)
# NEXT Lecture CMIT 2
# Advanced programming:  "def" for a function, "class" for an advanecd function, plot/imshow
#   plus one or two examples 

