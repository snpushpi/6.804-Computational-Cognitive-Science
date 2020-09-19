import numpy
import math
def calc(h,t):
    #theta^h*(1-theta)^t
    #theta ranges from 0.01 to 1
    sum = 0
    for theta in numpy.arange(0.01,1.01,0.01):
        elt = (theta**h)*((1-theta)**t)
        sum = sum + elt
    return sum*0.01
def logistic_func(a,x,b):
    exponent = -a*x+b
    return 1/(1+np.exp(exponent))
    
data = {1: {"H":3 ,'T':2,'s':5},
2: {'H':1, 'T':4,'s':5},
3: {'H':5,'T':0,'s':5},
4: {'H':4,'T':6,'s':10},
5 : {'H':8,'T':2,'s':10},
6: {'H':0,'T':10,'s':10},
7: {'H':10, 'T':15,'s':25},
8: {'H':20, 'T':5,'s':25},
9: {'H':25,'T':0,'s':25}}
result_dict_2 = {}
for i in range(1,10,1):
    result_dict_2[i]=calc(data[i]['H'],data[i]['T'])
print(result_dict_2)
#generate log posterior ratio
log_posterior_ratio = {}
for i in range(1,10,1):
    fraction = calc(data[i]['H'],data[i]['T'])/(2**data[i]['s'])
    log_posterior_ratio[i]= math.log(fraction)
print(log_posterior_ratio)



