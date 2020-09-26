import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
def calc(h,t):
    #theta^h*(1-theta)^t
    #theta ranges from 0.01 to 1
    sum = 0
    for theta in np.arange(0,1,0.01):
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

#generate log posterior ratio
log_posterior_ratio = {}
transformed_model = {}
for i in range(1,10,1):
    fraction = 1/((2**data[i]['s'])*result_dict_2[i])
    
    log_posterior_ratio[i]= math.log(fraction)+math.log(0.4/0.6)
    transformed_model[i]=logistic_func(1,log_posterior_ratio[i],0)

scaled_model = {}
for i in range(1,10):
    scaled_model[i]=7-6*transformed_model[i]
print(scaled_model)
human_data_c11 = {1:1, 2:3, 3:4, 4:1, 5:4, 6:6, 7:2, 8:5, 9:7}
human_data_c12 = {1:1, 2:2, 3:5, 4:1, 5:4, 6:6, 7:1, 8:5, 9:7 }
human_data_c11_a = {1:1, 2:2.5, 3:4.5, 4:1, 5:4, 6:6, 7:1.5, 8:5, 9:7}
human_data_c21 = {1:1, 2:2, 3:3, 4:1, 5:3, 6:5, 7:1, 8:3, 9:7 }
human_data_c22 = {1:1, 2:2, 3:5, 4:1, 5:3, 6:6, 7:2, 8:4, 9:7 }
human_data_c22_a = {1:1, 2:2, 3:4, 4:1, 5:3, 6:5.5, 7:1.5, 8:3.5, 9:7 }
l_scaled_model = []
l_human_data_c11 = []
l_human_data_c12 = []
l_human_data_c21 = []
l_human_data_c22 = []
l_human_data_c11_a = []
l_human_data_c22_a = []
for i in range(1,10):
    l_scaled_model.append(scaled_model[i])
    l_human_data_c11.append(human_data_c11[i])
    l_human_data_c12.append(human_data_c12[i])
    l_human_data_c21.append(human_data_c21[i])
    l_human_data_c22.append(human_data_c22[i])
    l_human_data_c11_a.append(human_data_c11_a[i])
    l_human_data_c22_a.append(human_data_c22_a[i])
from scipy.stats import pearsonr
# calculate Pearson's correlation
corr11, _ = pearsonr(l_scaled_model, l_human_data_c11)
corr12,_ = pearsonr(l_scaled_model, l_human_data_c12)
corr21,_ = pearsonr(l_scaled_model, l_human_data_c21)
corr22,_ = pearsonr(l_scaled_model, l_human_data_c22)
corr1, _ = pearsonr(l_scaled_model, l_human_data_c11_a)
corr2, _ = pearsonr(l_scaled_model, l_human_data_c22_a)
print(corr1, corr2)
epsilon_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
fig = plt.figure()
fig.patch.set_facecolor('xkcd:white')
plt.clf()
plt.xlabel('ith coin sequence')
plt.ylabel('Rating or scaled value from model')
plt.plot(epsilon_list, l_human_data_c11,'b-', label = 'Ratings for cover story 1, person 1')
plt.plot(epsilon_list, l_human_data_c12,'g-', label = 'ratings for cover story 1, person 2')
plt.plot(epsilon_list, l_human_data_c21,'c-', label = 'Ratings for cover story 2, person 1')
plt.plot(epsilon_list, l_human_data_c22,'r-', label = 'Ratings for cover story 2, person 2')
plt.plot(epsilon_list, l_scaled_model,'y-', label = 'Scaled Model prediction')
plt.legend()
plt.show()


