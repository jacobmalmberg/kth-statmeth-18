import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.stats
import math
#import seaborn as sns

class SVParams:
	def __init__(self, phi, sigma, beta):
		self.phi = phi
		self.sigma = sigma
		self.beta = beta

def generator(T, sv_params):
	x = np.zeros(T)
	y = np.zeros(T)
	x[0] = np.random.normal(0, sv_params.sigma)
	y[0] = np.random.normal(0, math.sqrt(np.power(sv_params.beta, 2) * np.exp(x[0])))
	for t in range(1, T):
		x[t] = np.random.normal(sv_params.phi * x[t-1], sv_params.sigma)
		y[t] = np.random.normal(0, math.sqrt(np.power(sv_params.beta, 2) * np.exp(x[t])))
	return x, y

def sis(obs, num_particles, sv_params):
	T = len(obs)
	x = np.zeros([T, num_particles], dtype=np.float64)
	w = np.zeros([T, num_particles], dtype=np.float64)
	for t in range(T):
		for n in range(num_particles):
			if t == 0:
				x[t][n] = np.random.normal(0, sv_params.sigma)
				w[t][n] = scipy.stats.norm.pdf(obs[t], loc = 0, scale = math.sqrt(np.power(sv_params.beta,2, dtype=np.float64)*np.exp(x[t,n], dtype=np.float64)))
				#w[t][n]

			else:
				x[t][n] = np.random.normal(sv_params.phi * x[t-1][n], sv_params.sigma)
				w[t][n] = scipy.stats.norm.pdf(obs[t], loc = 0, scale = math.sqrt(np.power(sv_params.beta,2)*np.exp(x[t,n])))* w[t-1][n]
				#w[t][n] = scipy.stats.norm.pdf(obs[t], loc = 0, scale = math.sqrt(np.power(sv_params.beta,2, dtype=np.float64)*np.exp(x[t,n], dtype=np.float64)))* w[t-1][n]
				#w[t][n] = np.multiply(scipy.stats.norm.pdf(obs[t], loc = 0, scale = np.sqrt(np.multiply(np.power(sv_params.beta,2, dtype=np.float64), np.exp(x[t,n], dtype=np.float64), dtype=np.float64), dtype=np.float64)), w[t-1][n], dtype=np.float64)
	return w, x

def smc_multinomial(obs, num_particles, sv_params):

	"""seyongs algo"""
	T = len(obs)
	x = np.zeros([T, num_particles])
	w = np.zeros([T, num_particles])
	for t in range(T):
		for n in range(num_particles):
			if t == 0:
				x[t][n] = np.random.normal(0, sv_params.sigma)
				w[t][n] = scipy.stats.norm.pdf(obs[t], loc = 0, scale = math.sqrt(np.power(sv_params.beta,2)*np.exp(x[t,n])))
			else:
				sampled_value = np.argmax(np.random.multinomial(1, w[t-1]))
				x[t][n] = np.random.normal(sv_params.phi * x[t-1][sampled_value], sv_params.sigma)
				w[t][n] = scipy.stats.norm.pdf(obs[t], loc = 0, scale = math.sqrt(np.power(sv_params.beta,2)*np.exp(x[t,n])))* w[t-1][sampled_value]

		#normalize
		w[t] = w[t] /np.sum(w[t])
	return w, x

def smc_stratified(obs, num_particles, sv_params):

	"""seyongs algo, modifierad för stratifiering"""
	"""https://pdfs.semanticscholar.org/f62a/0536c7e8395055f2072 sid 6
	https://www.cs.ubc.ca/~arnaud/samsi/samsi_lec3.pdf slide 34 """

	T = len(obs)
	x = np.zeros([T, num_particles])
	w = np.zeros([T, num_particles])

	u1 = np.random.uniform(low=0, high = 1/num_particles)
	u = np.array([u1 + (1/num_particles)*i for i in range(num_particles)])

	for t in range(T):
		if t > 0:
			#total count of kids
			count = 0

		for n in range(num_particles):
			if t == 0:
				x[t][n] = np.random.normal(0, sv_params.sigma)
				w[t][n] = scipy.stats.norm.pdf(obs[t], loc = 0, scale = math.sqrt(np.power(sv_params.beta,2)*np.exp(x[t,n])))
			else:
				lower = np.sum(w[t-1][:n])
				higher = np.sum(w[t-1][:n+1])
				nr_children = np.sum((lower <= u) & (u < higher))
				if nr_children != 0:
					for i in range(nr_children):
						x[t][i+count] = np.random.normal(sv_params.phi * x[t-1][n], sv_params.sigma)
						w[t][i+count] = scipy.stats.norm.pdf(obs[t], loc = 0, scale = math.sqrt(np.power(sv_params.beta,2)*np.exp(x[t,i+count])))* w[t-1][n]
					count += nr_children
		#normalize
		w[t] = w[t] /np.sum(w[t])
	return w, x

def normalize_weights(w, T):
	"""NOTERA T-1"""
	w_norm = np.divide(w[T-1], np.sum(w[T-1]))
	return w_norm

def mean_squared(x_vec, x_true):#, function):
	diff = x_true - x_vec
	mse = (np.square(diff)).mean()
	#print("MSE for " + function.__name__ + " " +str(mse))
	return mse

def multiple_runs(function):
	num_particles_list = [100, 500, 1000, 1500]# 2000, 2500, 3000]
	#samples_list = list()
	for num_particles in num_particles_list:
		seed = 57832 #prova seed 1
		np.random.seed(seed)
		T = 100
		params = SVParams(1, 0.16, 0.64)
		x_truth, y = generator(T, params)
		print("particle numbers: "+str(num_particles)+"\n")
		w, x1 = function(y, num_particles, params)
		#w_norm = normalize_weights(w, T)
		mean_squared(x1, x_truth[-1], function)
		#x_hatT = np.sum(np.multiply(w_norm, x1[T-1]))
		#samples_list.append(x1)

def mse(x_vecT, x_truthT, w_vecT, num_particles):
	diff = np.power(x_vecT - x_truthT, 2)
	product = np.multiply(w_vecT, diff)
	sum = np.sum(product)
	# sum=0
	# for i in range(num_particles):
	# 	sum+=(w_vecT[i]*(x_vecT[i] - x_truthT)**2)
	return sum

def mean_estimate(function):
	num_particles_list = [100, 200, 300, 400]# 2000, 2500, 3000]
	num_particles_list = [200, 400, 600]#, 800, 1000]

	for num_particles in num_particles_list:
		seed = 11111##57832 #prova seed 1
		np.random.seed(seed)
		T = 100
		params = SVParams(1, 0.16, 0.64)
		x_truth, y = generator(T, params)
		print("particle numbers: "+str(num_particles)+"\n")
		w, x1 = function(y, num_particles, params)
		w_norm = normalize_weights(w, T)
		mse1=mse(x1[-1], x_truth[-1], w_norm, num_particles)
		print(mse1)
		print("---")

def printer(function, x, w, T):
	"""prints xhat and var"""

	#w_norm = normalize_weights(w, T)
	w_norm= w[T-1] /np.sum(w[T-1])
	x_hatT = sum(x[T-1]*w_norm)
	empirical_var = np.var(w_norm)
	print('\n')
	print(function.__name__)
	print('empirical_var weights in last timestep: ', empirical_var)
	print('x_hatT: ', x_hatT)

def weighted_particle_plot(x_truth_vec, x_vec, w_vec, T, function, seed):
	"""plottar en viktad mean av partiklarna per timestep"""

	x_mean_vec=list()
	for t in range(T):
		w_norm = w_vec[t] / np.sum(w_vec[t])
		mean=np.sum(x_vec[t]*w_norm)
		x_mean_vec.append(mean)

	plt.plot(x_mean_vec, '-r', label='mean of particles')
	plt.plot(x_truth_vec, label='truth')
	plt.legend(loc='upper left')
	plt.title(function.__name__ +" seed " +str(seed))
	plt.show()

def time(allOfIt):
	# om man vill timea ngt. skicka in hela funktionsanropet som en sträng
	import timeit
	t1=timeit.Timer(stmt=allOfIt)
	print(allOfIt +" tid " +str(t1.timer()))

def plot_hist(data, x_label, y_label, text_labels):
	sns.set()
	# for datat, text_label in zip(data, text_labels):
	# 	_ = plt.hist(datat, label = text_label)
	_ = plt.hist(data, label = text_labels, bins = [0,0.005,0.01,0.015,0.02,0.025])
	_ = plt.xlabel(x_label)
	_ = plt.ylabel(y_label)
	_ = plt.legend(loc='upper right')
	plt.show()

def main():
	seed = 57832 #prova seed 1
	np.random.seed(seed)
	T = 1000
	params = SVParams(1, 0.16, 0.64)
	x_truth, y = generator(T, params)

	num_particles = 100
	#plt.ion()

	print("X_truth", x_truth[T-1])
	print("T", T)
	print("num_particles", num_particles)
	print("seed", seed)

	data = list()
	w1, x1 = sis(y, num_particles, params)
	data.append(w1[-1])
	#multiple_runs(sis)
	#mean_estimate(smc_stratified)

	#m1 = mean_squared(x1, x_truth[-1])
	#print('m1: ', m1)
	#input()
	w2, x2=smc_multinomial(y, num_particles, params)
	data.append(w2[-1])
	#m2 = mean_squared(x2, x_truth[-1])
	#print('m2: ', m2)
	#input()

	w3, x3 =smc_stratified(y, num_particles, params)
	data.append(w3[-1])
	#m3 = mean_squared(x3, x_truth[-1])
	#print('m3: ', m3)
	#input()
	#
	x_label = 'The bins of normalized weights'
	y_label = 'Number of occurences in each bin'
	text_labels = ['Sis','Multinomial','Stratified']
	#
	plot_hist(data, x_label, y_label, text_labels)
	# input('Stop!')

	printer(sis,x1,w1,T)
	printer(smc_multinomial,x2,w2,T)
	printer(smc_stratified,x3,w3,T)

	# #
	# weighted_particle_plot(x_truth, x1, w1, T, sis, seed)
	# weighted_particle_plot(x_truth, x2, w2, T, smc_multinomial, seed)
	# weighted_particle_plot(x_truth, x3, w3, T, smc_stratified, seed)


if __name__ == '__main__':
	main()
