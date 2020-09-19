import numpy as np
import matplotlib.pyplot as plt



def number_game_simple_init(N, interval_prior, math_prior):

	if abs((interval_prior + math_prior) - 1) > 0.05:
	  raise ValueError('Sum of interval prior and math prior should be 1!')

	# generate interval concepts of small and medium length
	hypotheses = np.zeros((0, N))
	vals = np.arange(N) + 1

	for size in np.arange(0, 20):
		for start in np.arange(N-size):
			end = start + size
			interval = np.zeros(N)
			interval[start:end+1] = 1
			hypotheses = np.vstack([hypotheses, interval])

	last_interval_concept = hypotheses.shape[0]

	#put in odds
	concept = np.equal(np.mod(vals, 2), 1).astype(int)
	hypotheses = np.vstack([hypotheses, concept])

	#put in multiples of 2 to 10
	for base in np.arange(2,11):
		concept = np.equal(np.mod(vals, base), 0).astype(int)
		hypotheses = np.vstack([hypotheses, concept])

	last_hypothesis = hypotheses.shape[0]

	#compute prior probabilities
	priors = np.empty(last_hypothesis)
	priors[:last_interval_concept] = interval_prior/last_interval_concept
	priors[last_interval_concept:] = math_prior/(last_hypothesis-last_interval_concept)

	return hypotheses, priors



def number_game_likelihood(hypothesis, data):
	"""
		hypothesis is a logical (0 or 1) vector on N elements, where
		hypothesis[i] = 1 iff i is contained in the extension of the
		concept represented by hypothesis.

		data is, similarly, a logical vector where data[i] = 1 iff
		i is contained in the observed dataset.

		note that length(hypothesis) == length(data) unless the caller
		of this procedure messed up

		TODO: first check if data is consistent with the given hypothesis.

		if it isn't, P(D|H) = 0.

		TODO: under strong sampling WITH REPLACEMENT, every consistent hypothesis
		assigns probability 1/(#options) to each data draw.
	"""

	l1 = len(data)
	l2 = len(hypothesis)
	if l1!=l2:
		return -float('inf')
	#if any data element is not in hypothesis, then return, -float('inf')
	for i in range(l1):
		if data[i]==1 and hypothesis[i]==0:
			return -float('inf')
	total_elt_in_hyp = 0
	total_elt_in_data = 0
	for i in range(l1):
		if hypothesis[i]==1:
		    total_elt_in_hyp+=1
	for i in range(l1):
	    if data[i]==1:
		    total_elt_in_data+=1
	return -np.log(total_elt_in_hyp**total_elt_in_data)

def number_game_plot_predictions(hypotheses, priors, data1, data2, data3, data4):
    """
        hypotheses = a matrix whose columns are particular hypotheses,
    represented as logical vectors reflecting datapoint membership

    priors = a vector of prior probabilities for each hypothesis

    data = a vector of observed numbers
    """
    hyps, N = hypotheses.shape
    def numbers_to_logical(data):
        if np.isscalar(data): data = [data]
        logical_data = np.zeros(N)
        for datum in data:
            logical_data[datum-1] = 1
        return logical_data
    logical_data1 = numbers_to_logical(data1)
    logical_data2 = numbers_to_logical(data2)
    logical_data3 = numbers_to_logical(data3)
    logical_data4 = numbers_to_logical(data4)
    
	# compute the posterior for every hypothesis
    def posterior_gen(logical_data):
        posteriors = np.zeros(hyps)
        for h in np.arange(hyps):
            log_joint = np.log(priors[h]) + number_game_likelihood(hypotheses[h,:], logical_data)
            joint = np.exp(log_joint)
            posteriors[h] = joint
        return posteriors
    posteriors1 = posterior_gen(logical_data1)
    posteriors2 = posterior_gen(logical_data2)
    posteriors3 = posterior_gen(logical_data3)
    posteriors4 = posterior_gen(logical_data4)
    posteriors1 /= np.sum(posteriors1)
    posteriors2 /= np.sum(posteriors2)
    posteriors3 /= np.sum(posteriors3)
    posteriors4 /= np.sum(posteriors4)

	# compute the predictive contribution for each
	# hypothesis and add it in to the predictive

    predictive1 = np.dot(posteriors1, hypotheses)
    predictive2 = np.dot(posteriors2, hypotheses)
    predictive3 = np.dot(posteriors3, hypotheses)
    predictive4 = np.dot(posteriors4, hypotheses)
	# plot it as a bar chart, also plot human data (if available)
	# and the top 6 hypotheses in decreasing order of posterior
	# probability

    fig, ax = plt.subplots(6,1, figsize=(7, 7))
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.85,
		left=0.05, right=0.95)

    ax[0].bar(np.arange(N)+1.0, predictive1, 0.5, color='k')
    if np.isscalar(data1): data1 = [data1]
    ax[0].set_title('Predictions given observation(s) %s'
		% ', '.join(str(d) for d in data1))
    ax[0].set_xlim([-0.5, (N+1)+0.5])
    ax[0].set_ylim([-0.05, 1.05])
    ax[1].bar(np.arange(N)+1.0, predictive2, 0.5, color='k')

    if np.isscalar(data1): data2 = [data2]
    ax[1].set_title('Predictions given observation(s) %s'
		% ', '.join(str(d) for d in data2))
    ax[1].set_xlim([-0.5, (N+1)+0.5])
    ax[1].set_ylim([-0.05, 1.05])
    ax[2].bar(np.arange(N)+1.0, predictive3, 0.5, color='k')

    if np.isscalar(data1): data3 = [data3]
    ax[2].set_title('Predictions given observation(s) %s'
		% ', '.join(str(d) for d in data3))
    ax[2].set_xlim([-0.5, (N+1)+0.5])
    ax[2].set_ylim([-0.05, 1.05])
    ax[3].bar(np.arange(N)+1.0, predictive4, 0.5, color='k')

    if np.isscalar(data1): data4 = [data4]
    ax[3].set_title('Predictions given observation(s) %s'
		% ', '.join(str(d) for d in data4))
    ax[3].set_xlim([-0.5, (N+1)+0.5])
    ax[3].set_ylim([-0.05, 1.05])

    plt.show()
hypotheses, prior = number_game_simple_init(100, 0.5, 0.5)
number_game_plot_predictions(hypotheses, prior, [80],[80,10],[80,10,60],[80,10,60,30])

