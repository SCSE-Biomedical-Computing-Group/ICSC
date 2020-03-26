from __future__ import division, absolute_import, print_function
import sys
if sys.version_info < (3,):
    range = xrange
import os
import numpy as np
from numpy.random import *  # for random sampling
from sklearn import metrics, cluster, preprocessing
import math
from copy import deepcopy
import random
from multiprocessing import Pool
import warnings
from kneed import KneeLocator
from pylab import *  # for plotting
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Array is not symmetric, and will be converted")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
#warnings.filterwarnings("ignore", message="No knee/elbow found")
warnings.filterwarnings("ignore", message="/usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:746: FutureWarning: The behavior of AMI will change in version 0.22.")

def get_threshold(adj_matrix, percent_threshold):
	'''
	Inputs
	adj_matrix (np.array (N, N)): the adjacency matrix of the network whose threshold is to be found
	percent_threshold (int): the percentage of edges to retain 
	Returns
	threshold (float): the boundary value below which the edges can be removed.
	'''
	sorted_graph_data = np.sort(adj_matrix, axis=None)
	sorted_graph_data = np.array(sorted_graph_data[sorted_graph_data>0])
	#print (sorted_graph_data)
	boundary_element_index = math.floor(sorted_graph_data.size*(100-percent_threshold)/100)
	threshold = sorted_graph_data[int(boundary_element_index)]
	#print (sorted_graph_data.size, boundary_element_index, threshold)
	return threshold


def get_consensus_matrix(partition, num_nodes):	
	'''
	Inputs
	partition (dict): key is the node id and the value is the assigned module label
	num_nodes (int): number of nodes in th network
	Returns
	consensus_matrix (2D list shape (num_nodes, num_nodes)): when two nodes i,j are in the same module consensus_matrix[i, j] = 1, else 0
	'''
	consensus_matrix = np.zeros((num_nodes, num_nodes))
	for key in partition:
		for node in partition:
			if partition[key] == partition[node]:
				consensus_matrix[key][node], consensus_matrix[node][key] = 1, 1

	return consensus_matrix

def get_elbow(mode, adj_matrix, max_modules, min_modules):
	'''
	Inputs
	mode (str): just a string label to keep track of what matrix we're initializing. No computational value. 
	adj_matrix (np.array (N, N)): the adjacency matrix of the network
	max_modules (int): Max number of expected labels
	min_modules (int): Min number of expected labels
	Returns
	elbow/knee (int): The most appropriate number of knee/elbow value
	'''
	s, principal_axes = np.linalg.eig(adj_matrix)
	N = max_modules + 1
	ind = np.arange(min_modules, N, 1)    # the x locations for the groups
	kn = KneeLocator(ind, s[min_modules:N], S=1.0, curve='convex', direction='decreasing', online=True)
	
	'''
	plt.figure()
	plt.xlabel('k')
	plt.ylabel('Distortion')
	plt.title('The Elbow Method showing the optimal k')
	plt.plot(ind, s[min_modules:N], 'bx-')
	plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	savefig('modules_' + str(mode) + '.eps', bbox_inches='tight', format='eps', dpi=200)
	plt.close()
	'''

	if kn.knee is None:
		return int((max_modules + min_modules)/2)
	return kn.knee

def get_initial_modularizations(prefix, adj_matrix, num_nodes, max_modules, min_modules):
	'''
	Inputs
	prefix (str): just a string label to keep track of what matrix we're initializing. No computational value. 
	adj_matrix (np.array (N, N)): the adjacency matrix of the network
	num_nodes (int): number of nodes in th network 
	max_modules (int): Max number of expected labels
	min_modules (int): Min number of expected labels

	Returns
	all_labels (dict): key is the number of modules and value is the corresponding modularization
	consensus_matrix (2D list shape (num_nodes, num_nodes)): when two nodes i,j are in the same module consensus_matrix[i, j] = 1, else 0
	best_labels (1D list): best modularization for the given adj_matrix
	'''

	all_labels = dict()

	for num_cluster in range(min_modules, max_modules, 1):
		random_state = randint(1, 100)    # Pick a random number between 1 and 100.
		all_labels[num_cluster] = cluster.SpectralClustering(n_clusters= num_cluster, random_state = random_state, n_init = 100, affinity='precomputed', assign_labels='discretize').fit_predict(adj_matrix)

	best_num_cluster = get_elbow(prefix, adj_matrix, max_modules, min_modules)

	partition = dict()
	best_labels = all_labels[best_num_cluster]

	for node, label in enumerate(best_labels):
		partition[node] = label
	
	consensus_matrix = get_consensus_matrix(partition, num_nodes)

	return all_labels, consensus_matrix, best_labels

def adjust_modular_partition(individual_all_modules, group_consensus_labels, individual_orig_labels, improvement_threshold, base_consensus_cost, modularisation_method, max_modules, min_modules):
	'''
	Inputs
	individual_all_modules (dict): key is the number of modules and value is the corresponding modularization for the given subject
	group_consensus_labels (1D list): group-level modularization
	individual_orig_labels (1D list): original individual modularization
	improvement_threshold (float): do not consider an improvement in modularization, if the improvement in consensus-cost in comparison to base_consensus_cost is below the improvement_threshold
	base_consensus_cost (float): original consensus cost.
	modularisation_method: currently only 'Spectral' is supported.
	max_modules (int): Max number of expected labels
	min_modules (int): Min number of expected labels
	
	Returns
	best_new_labels (1D list): new modularization with highest similarity to the group_consensus_labels
	best_consensus_cost (float): the new consensus cost (if changed)
	modify (boolean): True if the modularization is modified

	'''
	modify = False
	best_consensus_cost = base_consensus_cost
	best_new_labels = individual_orig_labels

	if modularisation_method == 'Spectral':
		for num_module in range(min_modules, max_modules, 1):
			labels = individual_all_modules[num_module]
			new_consensus_cost = metrics.adjusted_mutual_info_score(group_consensus_labels, labels)

			if new_consensus_cost > best_consensus_cost and (new_consensus_cost - base_consensus_cost) > improvement_threshold:
				best_consensus_cost = new_consensus_cost 
				modify = True
				best_new_labels = labels
	else:
		print ('Modularisation Method not Supported!')

	return best_new_labels, best_consensus_cost, modify