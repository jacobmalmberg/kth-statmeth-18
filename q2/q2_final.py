# coding=utf-8
from q2generator import *
import numpy as np
import pdb
import math
import copy
from collections import Counter
import matplotlib.pyplot as plt

nr_accepted = 0
count = 0

def matrix_to_list(X):
	"""Converts
	[[3, 2, 2], [2, 1, 2], [2, 3, 3]] -> [3, 2, 2, 2, 1, 2, 2, 3, 3]
	"""
	return np.array(X).flatten().tolist()

def list_to_matrix(X):
	"""Converts
	[3, 2, 2, 2, 1, 2, 2, 3, 3] -> [[3, 2, 2], [2, 1, 2], [2, 3, 3]]
	"""
	theList=list(X)
	return [X[i:i+3] for i in range(0, len(X), 3)]

def conditional_likelihood(o, G, start, X, p):
	""" Computes conditional likelihood
	returns the logged prob
	"""
	T = len(o)
	#p = 0.1 #hardcoded

	product=0 #vad vi räknar ut

	#next_node = start.get_switch_dir() # ger pos (0,2)
	#print(G_truth[1][1].equals_0_dir(G_truth[1][2]))
	s_t = G.get_next_node(start, 0, X)[0] ##anta att vi följer switchdir

	##vilken siffra vi kom in i
	entrance = G.get_entry_direction(start,s_t)

	#print("start", start)
	#print("---")



	for t in range(1, T):

		if entrance == 0:
			#kommer in i en nolla
			if o[t] == X[s_t.row][s_t.col]:
				emission_prob=1-p
			else:
				emission_prob=p

			#följ switch setting
			next = G.get_next_node(s_t, entrance, X)[0]
			#next = get_switch_dir_node(G,s_t.get_switch_dir())
			entrance = G.get_entry_direction(s_t,next)
			#zero_entrance = next.equals_0_dir(s_t)
			s_t=next

		else:
			#kommer inte in i en nolla
			if o[t] == 0:
				emission_prob=1-p
			else:
				emission_prob=p

			#följ inte switch setting
			next = G.get_next_node(s_t, entrance, X)[0]
			#next = get_switch_dir_node(G, s_t.get_0_dir())
			#zero_entrance = next.equals_0_dir(s_t)
			entrance = G.get_entry_direction(s_t,next)
			s_t=next

		product+=math.log(emission_prob)

		#print ("entrance", entrance)
		#print("obs",o[t])
		#print("emission_prob", emission_prob)
		#print("product", math.exp(product))
		#print("---")
		#input("sdf")
	return product


def compute_s1(s1, G, o, X, error_prob):
	nominator = conditional_likelihood(o, G, s1, X, error_prob) + math.log(1/9)
	#(o, G, start, X, p):
	return nominator


def gibbs_update_s1(o, X, G, error_prob):
	"""Q4: Gibbs update for s1
	"""
	#s1 is a start node

	#dist=list()
	prob_list=list()

	for r in range(G.lattice_size):
		for c in range(G.lattice_size):
			s1=G.get_node(r,c)
			#print(s1)
			#print(G)
			prob = compute_s1(s1, G, o ,X, error_prob)
			prob_list.append(prob)

	#now normalize
	prob_array=np.array(prob_list)

	#normalize
	p = np.exp(prob_array - np.max(prob_array))
	p = p/ np.sum(p)

	#sample new s1
	s1_new = np.argmax(np.random.multinomial(1, p))
	s1_new=G.get_node(s1_new//G.lattice_size, s1_new%G.lattice_size)
	#print('p: ', p)
	#print('s1_new: ', s1_new)
	#input()
	return s1_new


def get_random():
	return (np.random.randint(low = 0, high = 3), np.random.randint(low = 0, high = 3))




# o: observations
# n_lattice: size of the lattice
# num_iter: number of MCMC iterations
def mh_w_gibbs(o, G, num_iter, error_prob=0.1):
	prob_res=[]

	s = [] # store samples for the start positions
	X = [] # store switch states
	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]
	for n in range(num_iter):
		s1 = gibbs_update_s1(o, X[-1], G, error_prob)
		s.append(s1)
		#print("orginal", X[-1])
		X_new = [[X[-1][y][x] for x in range(G.lattice_size)] for y in range(G.lattice_size)]

		for i in range(G.lattice_size):
				for j in range(G.lattice_size):
					proposed_sample = np.random.randint(low = 1, high = 4)
					denominator = math.exp(conditional_likelihood(o, G, s1, X_new, error_prob))

					temp_x = [[X_new[y][x] for x in range(G.lattice_size)] for y in range(G.lattice_size)]
					#temp_x = list(list())
					#for a in range(G.lattice_size):
						#for b in range(G.lattice_size):
							#temp_x[a][b] = X_new[a][b]
					#temp_x = copy.deepcopy(X_new)
					#index= get_random()
					temp_x[i][j] = proposed_sample
					#print('proposed_sample: ', proposed_sample)
					#print('X_new: ', X_new)
					#print('temp_x: ', temp_x)
					#input()
					nominator = math.exp(conditional_likelihood(o, G, s1, temp_x, error_prob))
					# check accepteance
					u = np.random.rand()
					#print('u: ',u)
					#print('nominator: ', nominator)
					#print('denominator: ', denominator)
					#print('kvoten: ',nominator/denominator)
					#input()
					global count
					count += 1
					if u < min(nominator/denominator,1):
						#print('funkar')
						X_new = [[temp_x[y][x] for x in range(G.lattice_size)] for y in range(G.lattice_size)]
						prob_res.append(nominator)
						global nr_accepted
						nr_accepted += 1
					#print(nominator)
					#input()
		X.append(X_new)
	return s, X


def gibbs(o, G, num_iter, error_prob=0.1):
	s = [] # store samples for the start positions
	X = [] # store switch states

	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]
	for n in range(num_iter):
		temp_x = [[X[-1][y][x] for x in range(G.lattice_size)] for y in range(G.lattice_size)]
		s1 = gibbs_update_s1(o, temp_x, G, error_prob)
		s.append(s1)

		for r in range(G.lattice_size):
			for c in range(G.lattice_size):
				#theNode=G.get_node(r,c)
				prob_list=list()

				## for every node try all switch settings
				for i in range(1,4):
					temp_x[r][c] = i
					prob=conditional_likelihood(o, G, s1, temp_x, error_prob)
					prob_list.append(prob)
				prob_array=np.array(prob_list)

				#normalize
				p = np.exp(prob_array - np.max(prob_array))
				p = p/ np.sum(p)

				#sample!
				temp_x[r][c] = np.argmax(np.random.multinomial(1, p)) +1
				#print(temp_x[r][c])

		X.append(temp_x)

	return s, X

# def block_gibbs(o, G, num_iter, error_prob=0.1):
# 	s = [] # store samples for the start positions
# 	X = [] # store switch states
# 	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
# 	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]
# 	for n in range(num_iter):
# 		pass
# 	return s, X

def block_gibbs(o, G, num_iter, error_prob=0.1):
	s = [] # store samples for the start positions
	X = [] # store switch states

	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]
	for n in range(num_iter):
		temp_x = [[X[-1][y][x] for x in range(G.lattice_size)] for y in range(G.lattice_size)]
		s1 = gibbs_update_s1(o, temp_x, G, error_prob)
		#print(1)
		s.append(s1)
		temp_x=matrix_to_list(temp_x)
		blockA=[0,2,4]
		blockB=[3,5,7]
		blockC=[1,6,8]

		blocks=[blockA,blockB,blockC]

		for i,block in enumerate(blocks):
			#if len(np.array(temp_x).shape) > 1: ##då är det en matris - vill ha en lista
			#	temp_x=matrix_to_list(temp_x)

			prob_list=list()
			nodeA= block[0]
			nodeB= block[1]
			nodeC= block[2]
			for var1 in range(1, 4):
				for var2 in range(1, 4):
					for var3 in range(1, 4):
						#print("hej",temp_x)
						#print("block ", i)
						temp_x[nodeA] = var1
						temp_x[nodeB] = var2
						temp_x[nodeC] = var3

						temp_x=list_to_matrix(temp_x)
						#print(temp_x)
						prob = conditional_likelihood(o, G, s1, temp_x, error_prob)
						prob_list.append(prob)
						temp_x=matrix_to_list(temp_x)
						#print(temp_x)

			prob_array=np.array(prob_list)
			#normalize
			p = np.exp(prob_array - np.max(prob_array))
			p = p/ np.sum(p)

			#print(p)
			#input()
			sampled_value = np.argmax(np.random.multinomial(1, p))
			#print(sampled_value)


			# https://stackoverflow.com/questions/11316490/convert-a-1d-array-index-to-a-3d-array-index
			var1 = (sampled_value // (G.lattice_size*G.lattice_size)) + 1
			var2 = ((sampled_value  // G.lattice_size) % G.lattice_size) + 1
			var3 = (sampled_value % G.lattice_size) + 1
			temp_x[nodeA] = var1
			temp_x[nodeB] = var2
			temp_x[nodeC] = var3
			#print(temp_x)
			#input()


		temp_x=list_to_matrix(temp_x)

		X.append(temp_x)

	return s, X


def generator_jacob(s):
	data_new = list()
	for y in s:
		if y[0] == 0 and y[1] == 0:
			data_new.append(1)
		elif y[0] == 0 and y[1] == 1:
			data_new.append(2)
		elif y[0] == 0 and y[1] == 2:
			data_new.append(3)
		elif y[0] == 1 and y[1] == 0:
			data_new.append(4)
		elif y[0] == 1 and y[1] == 1:
			data_new.append(5)
		elif y[0] == 1 and y[1] == 2:
			data_new.append(6)
		elif y[0] == 2 and y[1] == 0:
			data_new.append(7)
		elif y[0] == 2 and y[1] == 1:
			data_new.append(8)
		elif y[0] == 2 and y[1] == 2:
			data_new.append(9)
	return data_new


def generator(s):
	data =[(s[x].row, s[x].col ) for x in range(len(s))]
	data_new = list()
	for y in data:
		if y[0] == 0 and y[1] == 0:
			data_new.append(1)
		elif y[0] == 0 and y[1] == 1:
			data_new.append(2)
		elif y[0] == 0 and y[1] == 2:
			data_new.append(3)
		elif y[0] == 1 and y[1] == 0:
			data_new.append(4)
		elif y[0] == 1 and y[1] == 1:
			data_new.append(5)
		elif y[0] == 1 and y[1] == 2:
			data_new.append(6)
		elif y[0] == 2 and y[1] == 0:
			data_new.append(7)
		elif y[0] == 2 and y[1] == 1:
			data_new.append(8)
		elif y[0] == 2 and y[1] == 2:
			data_new.append(9)
	return data_new


def plot_conv_rate(function, o, G, num_iter, p):

	"""
	skapa en plot för convergence rate"""

	s, X = function(o, G, num_iter, p) #28.8% #med burn-in 26%
	data =[(s[x].row, s[x].col ) for x in range(len(s))]
	bi=100 #burnin
	halva= len(data)//2

	data1 =data[halva:]
	data2 =data[:bi]
	#print('s_truthg[0]: ', s_truth[0])

	counter_post=Counter(data1)
	counter_bi = Counter(data2)

	data_new = generator_jacob(data1)
	data_new1 = generator_jacob(data2)


	print("counter efter burn-in på "+ str(bi) +" samples" + str(counter_post))


	print("counter burn-in", counter_bi)

	countlist = [Counter(data_new), Counter(data_new1)]
	printlist = [" for second half of chain", " for first " +str(bi) +" samples (burn-in)"]

	fig, axes = plt.subplots(2, figsize=(10,10))
	fig.subplots_adjust(hspace=0.3)
	i = 0
	for dict in countlist:
		print(dict)
		axes[i].bar(dict.keys(), dict.values(), color='r')
		axes[i].set_title(function.__name__ + printlist[i])
		axes[i].set_ylabel('Occurrences')
		axes[i].set_xlabel('Start positions on x-axis')
		i += 1
	plt.show()



def main():
	seed = 17
	n_lattice = 3
	T = 100
	p = 0.1
	G, X_truth, s_truth, o = generate_data(seed, n_lattice, T, p)
	#print(o)
	#print('X_truth: ', X_truth)

	# randomize the switch states -- get initial state

	X = sample_switch_states(n_lattice)

	# infer s[0] and switch states given o and G
	num_iter = 1000


	# for j in range(2):
	# 	np.random.seed(j)
	# 	s, X,prob_res = mh_w_gibbs(o, G, num_iter, p)
	# 	prob_vec = list()
	# 	for a,b in zip(s,X):
	# 		prob_vec.append(conditional_likelihood(o,G,a,b,p))
	# 	plt.plot(np.exp(prob_vec)[1::10])
	# plt.show()
	# input()

	s, X = block_gibbs(o, G, num_iter, p)

	s, X = mh_w_gibbs(o, G, num_iter, p) # 28.4% accuracy
	s, X = gibbs(o, G, num_iter, p) # 28.2% accuracy

	#plot_conv_rate(mh_w_gibbs, o, G, num_iter, p)
	mixing_rate = nr_accepted/count
	print(nr_accepted)
	print(count)
	input('Here comes the mixing rate')
	print('Mixing rate: ', mixing_rate)




	# # alexandras seed 225, 11, 2222
	# parameter_list = list()
	# seed_list = list()
	# for seed in range(4,16,4):
	# 	seed_list.append(seed)
	# 	np.random.seed(seed)
	#
	# 	s, X = gibbs(o, G, num_iter, p)
	# 	parameter_list.append(s)

	# s, X = mh_w_gibbs(o, G, num_iter, p)
	# data =[(s[x].row, s[x].col ) for x in range(len(s))]

	# input()
	# print(Counter(data))



	#print('s: ', s)

	#print('X: ', X[-1:])
	#print('X_truth: ', X_truth)


	#data = data[-100:]

	#print(Counter(data))


if __name__ == '__main__':
	main()
