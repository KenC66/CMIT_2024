# K Chen @ strath.ac.uk
# CMIT 2024 Python Lecture 2 == CMIT_P2.ipynb  [jupyter nbconvert --to python notebook.ipynb]
''' Content in L2  [To launch juypter, type "python -m jupyter notebook" or "juypter notebook"]
4) "def"   for defining a function  (like function in Matlab or proc in Maple)
5) "class" advanced version of "def" but essential for DL 
6) HW Exercise 2
'''
print('This is running: ', __file__.split('\\')[-1])
# $s2=\displaystyle\sum_{j=0}^n \frac{x^j}{j!}$ 
print('\nTask 9 --- re-do Task 6 summaton by "def')
import math

def my_sum(x, n):
    print('\n\t(1) for -- summation Task 6 by def (efficient)')
    s2 = 1;   t=x;     # Efficient code // "if" for less printing
    for j in range(1,n+1):
        if j % 5 == 0 or j==1 or j==n:
            print(j, t)
        s2 += t
        t = x*t/(j+1)       # note when j=1,  t=x^2 / 2!
    return s2

x = 0.12
s = my_sum(x, 12)
print('\nPartial Sum =', s, end=' ')
print('when x=', x, ',  s = %.8f' % s, 'NB exp(x) = %.8f' % math.exp(x), ' er={:.2e}\n'.format(s-math.exp(x)) )
print('-----------------------------------------------------------------------------')

# CASE 1 -- no input parameters
class my_test0():
    a = 1234
    def my_sum0(something,x, n):
        print('\n\t(2) for -- summation Task 6 by class 1 (efficient)')
        s2 = 1;   t=x;     # Efficient code // "if" for less printing
        for j in range(1,n+1):
            if j % 5 == 0 or j==1 or j==n:
                print(j, t)
            s2 += t
            t = x*t/(j+1)       # note when j=1,  t=x^2 / 2!
        return s2
x = 0.12
s = my_test0().my_sum0(x, 12)
print('\nPartial Sum =', s, end=' ')
print('when x=', x, ',  s = %.8f' % s, 'NB exp(x) = %.8f' % math.exp(x), ' er={:.2e}\n'.format(s-math.exp(x)) )
print('-----------------------------------------------------------------------------')

# CASE 2 -- outer input parameters  [NEW to MOST of US]
class our_test1:
    def __init__(somename, x=0.12, n=12):
        somename.x = x;    somename.n = n
    def my_sum1(somename):
        print('\n\t(3) for -- summation Task 6 by class 2 (efficient)')
        s2 = 1;   t=somename.x;     # Efficient code // "if" for less printing
        for j in range(1,somename.n+1):
            if j % 5 == 0 or j==1 or j==somename.n:
                print(j, t)
            s2 += t
            t = somename.x*t/(j+1)       # note when j=1,  t=x^2 / 2!
        return s2

sa = our_test1().my_sum1() # No input or default input used
sb = our_test1(x=0.12, n=12).my_sum1() # Supplied values now
sn = our_test1(x=0.52, n=15).my_sum1()
print('\nPartial Sum =', sa, 'or ', sb, 'vs NEW sn =', sn)
print('-----------------------------------------------------------------------------------')

# CASE 4 -- Inner and Outer (mixed) input parameters  [NEW to MOST of US]
class our_test3:
    def __init__(somename, x=0.12):
        somename.x = x    # not used
    def my_sum3(somename, n=12):
        print('\n\t(4) for -- summation Task 6 by class 3 (efficient)')
        s2 = 1;   t=somename.x;     # Efficient code // "if" for less printing
        for j in range(1,n+1):
            if j % 5 == 0 or j==1 or j==n:
                print(j, t)
            s2 += t
            t = somename.x*t/(j+1)       # note when j=1,  t=x^2 / 2!
        return s2

sa = our_test3().my_sum3() # No input or default input used
sb = our_test3(x=0.12).my_sum3( n=12) # Supplied values now
sn = our_test3(x=0.12).my_sum3( n=15)
print('\nPartial Sum =', sa, 'or ', sb, 'vs NEW sn =', sn)
print('-----------------------------------------------------------------------------------')

###########################Image Processing##############################
import glob
from skimage import io
import matplotlib.pyplot as plt

str_img1 = 'Cells/train/image/*.png' 
str_lab1 = 'Cells/train/label/*.png'  
files_1  = sorted(glob.glob( str_img1, recursive=True) ) # Images
files_2  = sorted(glob.glob( str_lab1, recursive=True) ) # Labels
Count = len(files_1)
Coun2 = len(files_2)
if not (Count == Coun2):
    print('Files may not match -- check')
print('Images files =', Count )
for i in range(Count):
    imgs = io.imread(files_1[i], 0)
    plt.imshow( imgs );  plt.title('Image %d' % i)
    plt.pause(0.1)

for i in range(Count):
    imgs = io.imread(files_1[i], 1)
    labs = io.imread(files_2[i], 0)
    plt.subplot(1,2,1);  plt.imshow( imgs );  plt.title('Image %d' % i)
    plt.subplot(1,2,2);  plt.imshow( labs );  plt.title('Label %d' % i)
    plt.pause(0.5)

# 
print('\nHome work 2 --- Generate a matrix and compute its condition number\n')

s='TASK:  Use the same Hilbert matrix as Home work 1.\n\
   Write a code using "class" to generate a matrix $A = H + sigma*I$ where\n\
   $H_{ij}=1/(i+j-1)$ for $i,j=1,...,n$,  sigama is a scalar and I=identity matrix.\n\
   Then compute its condition number $cond(A)$ for the cases of (i) sigma=0.5 and n=4\n\
   (ii) sigma=0.5 and n=8\n\
   If your code runs okay,  just show it quickly and your answers on screen in the next lecture\n'
print(s)

import datetime
time0 = datetime.datetime.now().strftime("%Y %m %d @ %H:%M:%S"); print('\t time =', time0)
# NEXT Lecture CMIT 3 -- How to run Python + DL on remote Google GPU via googledrive