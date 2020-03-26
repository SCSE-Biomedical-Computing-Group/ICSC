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

def get_threshold(graph_data, percent_threshold):
	'''
	Inputs
	Returns
	'''
	sorted_graph_data = np.sort(graph_data, axis=None)
	sorted_graph_data = np.array(sorted_graph_data[sorted_graph_data>0])
	#print (sorted_graph_data)
	boundary_element_index = math.floor(sorted_graph_data.size*(100-percent_threshold)/100)
	threshold = sorted_graph_data[int(boundary_element_index)]
	#print (sorted_graph_data.size, boundary_element_index, threshold)
	return threshold


def get_consensus_matrix(partition):	
	'''
	Inputs
	Returns
	'''
	num_nodes = 264;
	subject_mod_matrix = np.zeros((num_nodes, num_nodes))
	for key in partition:
		for node in partition:
			if partition[key] == partition[node]:
				subject_mod_matrix[key][node], subject_mod_matrix[node][key] = 1, 1;

	return subject_mod_matrix;

def get_elbow(mode, graph_data, max_labels, min_labels):
	s, principal_axes = np.linalg.eig(graph_data)
	N = max_labels + 1
	ind = np.arange(min_labels, N, 1)    # the x locations for the groups
	kn = KneeLocator(ind, s[min_labels:N], S=1.0, curve='convex', direction='decreasing', online=True)
	
	'''
	plt.figure()
	plt.xlabel('k')
	plt.ylabel('Distortion')
	plt.title('The Elbow Method showing the optimal k')
	plt.plot(ind, s[min_labels:N], 'bx-')
	plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	savefig('modules_' + str(mode) + '.eps', bbox_inches='tight', format='eps', dpi=200)
	plt.close()
	'''
		
	if kn.knee is None:
		return int((max_labels + min_labels)/2)
	return kn.knee

def get_initial_modularizations(prefix, subject_data, max_modules, min_modules):
	'''
	Inputs
	Returns
	'''
	all_labels = dict()

	for num_cluster in range(min_modules, max_modules, 1):
		random_state = randint(1, 100)    # Pick a random number between 1 and 100.
		all_labels[num_cluster] = cluster.SpectralClustering(n_clusters= num_cluster, random_state = random_state, n_init = 100, affinity='precomputed', assign_labels='discretize').fit_predict(subject_data)

	best_num_cluster = get_elbow(prefix, subject_data, max_modules, min_modules)

	partition = dict()
	best_labels = all_labels[best_num_cluster]

	for node, label in enumerate(best_labels):
		partition[node] = label
	
	consensus_matrix = get_consensus_matrix(partition)

	return all_labels, consensus_matrix, best_labels

def adjust_modular_partition(subject_all_modules, group_consensus_labels, subject_orig_labels, improvement_threshold, old_consensus_cost, modularisation_method, max_modules, min_modules):
	'''
	Inputs
	Returns
	'''
	modify = False
	best_new_ami = old_consensus_cost
	best_new_labels = subject_orig_labels

	if modularisation_method == 'Spectral':
		for num_module in range(min_modules, max_modules, 1):
			labels = subject_all_modules[num_module]
			new_consensus_cost = metrics.adjusted_mutual_info_score(group_consensus_labels, labels)

			if new_consensus_cost > best_new_ami and (new_consensus_cost - old_consensus_cost) > improvement_threshold:
				best_new_ami = new_consensus_cost 
				modify = True
				best_new_labels = labels
	else:
		print ('Modularisation Method not Supported!')

	return best_new_labels, best_new_ami, modify