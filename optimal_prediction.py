import numpy as np
import matplotlib.pyplot as plt
import scipy.stats




def opt_build_powerlaw_joint(tobs):
    t_obs = tobs
	#Observe that P(t_total) = width of smaller rectangle * probability density function at that point
    #or P(t_total) = width of smaller rectangle * (t_total)^(-gamma)
	#Now when we call this function's result, spcially thru opt_compute posterior, we have this width constant  in the numerator and 
	#denominator of that function in all terms, so basically they are cancelled out. So I this function, where we are supposed to return the
    #P(Theta, t_obs)= P(Theta)*P(t_obs|theta) and we will be using for theta = t_total later, we will still say P(theta)=P(t_total)=(t_total)^(-gamma)
	#because that will be cancelled anyway
	#Now we will return a function of theta here.
    def joint(theta):
    	if theta<t_obs:
    	    return 0
    	return theta**(-3.43)
    return joint





def opt_build_lifespan_joint(tobs):
	#Same logic here, though P(theta) is not equal probabiity density of that point, but we can still use it since it will be called 
	#from opt_compute_posterior where in numerators and denominator terms, we will have this term referring to the width , ie, p(theta)=width*
    #probability density which will be cancelled out from numerator and denominator
    def joint(theta):
    	if theta <tobs:
    		return 0
    	return scipy.stats.norm(75,16).pdf(theta)/theta

    return joint


#We will try for different observed values of t with this function here where theta or t_total ranges from 0 to 300
def opt_compute_posterior(joint, theta_min, theta_max, num_steps):
	"""
		Computes a table representation of the posterior distribution
		with at most num_steps joint density evaluations, covering the
		range from theta_min to theta_max.

		People interested in fancier integrators should feel free to
		modify the signature for this procedure, as well as its callers,
		as appropriate.

		TODO: compute Z along with an unnormalized table

		TODO: normalize joint

	"""
	#First get the thetavals vector
	thetavals = []
	diff = (theta_max-theta_min)/num_steps
	for i in np.arange(theta_min, theta_max+diff, diff):
		thetavals.append(i)
	Z = 0
	for theta in thetavals:
		Z+= joint(theta)
	print(Z)
	postvals = []
	for theta in thetavals:
		postvals.append(joint(theta)/Z)
	return thetavals, postvals

def opt_predictions_plot(function1, function2, theta_min, theta_max):
    observed_times = list(range(25,50,5))
    for observe in observed_times:
        thetavals, postvals = function1(function2(observe),theta_min,theta_max,300)
        fig = plt.figure()
        fig.patch.set_facecolor('xkcd:white')
        plt.clf()
        plt.ylabel('p(t_total|t_obs)')
        plt.xlabel('t_total')
        plt.plot(thetavals, postvals)
        plt.title('t_Obs = {}'.format(observe))
        plt.show()
		#generating medians
def opt_median_prediction(function1, function2, theta_min, theta_max):
    median  = []
    j = 0
    observed_times = list(range(25,50))
    for observe in observed_times:
        thetavals, postvals = function1(function2(observe), theta_min, theta_max, 300000)
        cum_thetavals = [postvals[0]]
        for i in range(1,len(thetavals)):
            cum_thetavals.append(cum_thetavals[-1]+postvals[i])
        while j<len(cum_thetavals):
            if cum_thetavals[j]>=0.5:
                median.append(thetavals[j])
                break
            j+=1
    fig = plt.figure()
    plt.plot(observed_times, median)
    plt.xlabel("t_obs")
    plt.ylabel(" Predicted median")
    plt.title("Median Prediction")
    plt.show()
#opt_predictions_plot(opt_compute_posterior, opt_build_powerlaw_joint, 0, 300)
opt_median_prediction(opt_compute_posterior, opt_build_powerlaw_joint, 0, 300)
