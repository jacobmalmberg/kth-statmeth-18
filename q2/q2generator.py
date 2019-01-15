import numpy as np
import pdb
import copy
import sys

def validate_idx(row, col, n):
	r = row
	c = col
	if row < 0:
		r = n-1
	elif row >= n:
		r = 0
	if col < 0:
		c = n-1
	elif col >= n:
		c = 0
	return (r, c)

class TrainLattice:
	def __init__(self, lattice_size):
		self.G = []
		self.lattice_size = lattice_size
		for r in range(self.lattice_size):
			row = []
			for c in range(self.lattice_size):
				node = Node(r, c)
				row.append(node)
			self.G.append(row)

	def get_node(self, r, c):
		return self.G[r][c]

	# returns next node
	def get_next_node(self, curr_node, entry_dir, X):
		exit_direction = 0 # if train entered curr_node through dir {1, 2, 3}, it exists through direction 0
		if entry_dir == 0: # otherwise, follows the current state of the switch
			exit_direction = X[curr_node.row][curr_node.col]
		(i, j) = curr_node.edges[exit_direction]
		(row, col) =  validate_idx(curr_node.row + i, curr_node.col + j, self.lattice_size)		
		return (self.get_node(row, col), exit_direction)

	def get_entry_direction(self, prev_pos, curr_pos):
		for direction in range(4):
			(i, j) = curr_pos.get_edge(direction)
			(r0, c0) = validate_idx(curr_pos.row + i, curr_pos.col + j, self.lattice_size)
			if r0 == prev_pos.row and c0 == prev_pos.col:
				return direction
		print("Bug")
		sys.exit(-1)

class Node:
	def __init__(self, row, col):
		self.row = row
		self.col = col
		self.sample_edge_labels()

	def sample_edge_labels(self):
		self.edges = [(-1, 0), (1, 0), (0, -1), (0, 1)]
		np.random.shuffle(self.edges) # shuffling the edges is equivalent to randomly assigning edge labels for this node

	def get_edge(self, i):
		return self.edges[i]

	def __repr__(self):
		ret = "\npos: " + str(self.row) + ", " + str(self.col) + "\n"
		for i in range(len(self.edges)):
			ret += str(i) + ": " + str(self.edges[i]) + "\n"
		return ret

# return random edge direction with error_prob or d[0] with (1 - error_prob)
def generate_single_obs(pos, edge_dir, p):
	u = np.random.uniform(0, 1, 1)[0]
	#pdb.set_trace()
	if u < p:
		print("Generating errorneous observation!")
		return np.random.randint(0, 4, 1)[0]
	return edge_dir

# s0: starting position
# X: switch state (as list of list)
# G: object of type TrainLattice
# error_prob: probability of observation error
# length: how many observations to generate
def simulate(s0, X, G, length, error_prob):
	s = [s0] # starting position
	o = []
	entry_directions = [0] # the train enters s_0 from edge direction 0
	for t in range(1, length):
		#pdb.set_trace()
		(s_t, exit_dir) = G.get_next_node(s[t-1], entry_directions[t-1], X)
		o_t = generate_single_obs(s_t, exit_dir, error_prob)
		entry_dir = G.get_entry_direction(s[t-1], s_t)
		s.append(s_t)
		entry_directions.append(entry_dir)
		o.append(o_t)
		#print(s[t-1], exit_dir)
		#print(s_t, entry_dir)
	return s, o

# returns a node selected at random
def sample_start_pos(G):
	r = np.random.randint(0, G.lattice_size)
	c = np.random.randint(0, G.lattice_size)
	s0 = G.get_node(r, c)
	return s0

# generate graph and switch states
# returns nxn matrix as list of list
def sample_switch_states(n):
	X = []
	for r in range(n):
		row = []
		for c in range(n):
			state = np.random.randint(low=1, high=4, size=1)[0]
			row.append(state)
		X.append(row)
	return X

def generate_data(seed, n_lattice, length, error_prob):
	np.random.seed(seed)
	G = TrainLattice(n_lattice)
	r = np.random.randint(0, n_lattice)
	c = np.random.randint(0, n_lattice)
	s0 = G.get_node(r, c)
	X = sample_switch_states(n_lattice)
	positions, observations = simulate(s0, X, G, length, error_prob)
	return G, X, positions, observations
