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
from utils import *

warnings.filterwarnings("ignore", message="Array is not symmetric, and will be converted")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="FutureWarning: The behavior of AMI will change in version 0.22.")

def single_run (params):
	run_id, directory, percent_threshold, individuals_list, max_labels, min_labels, num_nodes, dataset = params

	num_individuals = len(individuals_list)
	np.random.seed()
	random_state = randint(1, 100)    # Pick a random number between 1 and 100.
	consensus_cost = np.zeros(num_individuals)
	group_consensus_matrix  = np.zeros((num_nodes, num_nodes)) 
	all_individuals_all_L_k = dict()
	individual_labels = dict()
	subjects_data = dict()

	'''
	%%%%%%%%%%%%%%%%%%%%
	Initial Modularization Starts
	%%%%%%%%%%%%%%%%%%%%
	'''

	for count, individual in enumerate(individuals_list):
		subjects_data[individual] = np.zeros((num_nodes, num_nodes))
		graph = np.load(os.path.join(directory, individual))
		threshold = get_threshold(graph, percent_threshold)
		graph = graph*(graph>threshold)
		np.fill_diagonal(graph, 0)
		subjects_data[individual] = graph

		all_individuals_all_L_k[individual], individual_mod_matrix, individual_calc_labels = get_initial_modularizations(str(count), subjects_data[individual], num_nodes, max_labels, min_labels)
		group_consensus_matrix += individual_mod_matrix
		individual_labels[individual] = individual_calc_labels


	_, _, group_consensus_labels = get_initial_modularizations('group', group_consensus_matrix, num_nodes, max_labels, min_labels)

	'''
	%%%%%%%%%%%%%%%%%%%%
	Initial Modularization Done
	%%%%%%%%%%%%%%%%%%%%
	'''

	for count, individual in enumerate(individuals_list):
		consensus_cost[count] = metrics.adjusted_mutual_info_score(individual_labels[individual], group_consensus_labels)

	group_consensus_matrix = group_consensus_matrix*(group_consensus_matrix>get_threshold(group_consensus_matrix, percent_threshold))
	np.fill_diagonal(group_consensus_matrix, 0)

	consensus_cost_threshold = 0.01 #minimum change in AMI in order to update the partition 
	iteration = 0
	number_individuals_adjusted = num_individuals
	
	with open(SAVE_DIR + 'ICSC_group_level_iter_' + str(iteration) + '.csv', 'ab') as out_stream:
		np.savetxt(out_stream,  [np.append(np.array([run_id, np.mean(consensus_cost), 0, np.unique(group_consensus_labels).size]), group_consensus_labels)],  delimiter=", ") #'run_id, Consensus Cost, Num of subjects adjusted, number of unique group labels, group labels 

	'''
	%%%%%%%%%%%%%%%%%%%%
	Iterative Refinement
	%%%%%%%%%%%%%%%%%%%%

	'''

	print ('Iteration ', iteration, ' consensus cost: ', np.mean(consensus_cost), 'number of subjects with changed modules', number_individuals_adjusted)

	while number_individuals_adjusted > 0:
		
		group_consensus_labels = np.zeros(num_nodes)
		
		L = get_elbow('group', group_consensus_matrix, max_labels, min_labels)
		group_consensus_labels = cluster.SpectralClustering(n_clusters= L, random_state = random_state, n_init = 100, affinity='precomputed', assign_labels='discretize').fit_predict(group_consensus_matrix)
		
		for count, individual in enumerate(individuals_list):
			consensus_cost[count] = metrics.adjusted_mutual_info_score(individual_labels[individual], group_consensus_labels)
	

		number_individuals_adjusted = 0
		group_consensus_matrix = np.zeros((num_nodes, num_nodes))

		for count, individual in enumerate(individuals_list):
			labels, new_consensus_cost, modify = adjust_modular_partition(all_individuals_all_L_k[individual], group_consensus_labels, individual_labels[individual], consensus_cost_threshold, consensus_cost[count], 'Spectral', max_labels, min_labels)
			individual_labels[individual] = labels

			if modify == True:
				number_individuals_adjusted +=1

			consensus_cost[count] = new_consensus_cost
			partition = dict()
			
			for node, label in enumerate(individual_labels[individual]):
				partition[node] = label
			
			individual_mod_matrix = get_consensus_matrix(partition, num_nodes)
			group_consensus_matrix += individual_mod_matrix
		
		iteration += 1

		print ('Iteration ', iteration, 'number of group level modules: ', L, ' consensus cost: ', np.mean(consensus_cost), 'number of subjects with changed modules', number_individuals_adjusted)

		'''
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		Saving the results from every iteration
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		'''

		'''
		%
		The files ICSC_group_level_iter_i.csv save the following information for every run and iteration i:
		run_id, consensus cost (for the iteartion i), num of subjects whose modularizations were changed, number of unique group level labels, group-level labels for each node
		%

		'''
		with open(SAVE_DIR + 'ICSC_group_level_iter_' + str(iteration) + '.csv', 'ab') as out_stream: 
			np.savetxt(out_stream,  [np.append(np.array([run_id, np.mean(consensus_cost),  number_individuals_adjusted, np.unique(group_consensus_labels).size]), group_consensus_labels)],  delimiter=", ") #'run_id, AMI, N_Cut, Num of subjects adjusted, labels 

		if number_individuals_adjusted/num_individuals <= 0: #Convergence
			with open(SAVE_DIR + 'ICSC_group_level_final_iter.csv', 'ab') as out_stream:
				np.savetxt(out_stream,  [np.append(np.array([run_id, iteration, np.mean(consensus_cost), number_individuals_adjusted, np.unique(group_consensus_labels).size]), group_consensus_labels)],  delimiter=", ") #'run_id, AMI, N_Cut, Num of subjects adjusted, labels


			'''
			%from utils import *
			Saving group consensus matrix for each run
			%
			'''
			np.savetxt(SAVE_DIR + 'group_consensus_matrix_run_' + str(run_id) + '.csv', group_consensus_matrix,  delimiter=", ")

			'''
			%
			The file ICSC_subject_labels_run_i.csv saves the following information for every run:
			subject_id, individual-level labels for each node 
			%
			'''
			for count, individual in enumerate(individuals_list):
				with open(SAVE_DIR + 'ICSC_subject_labels_run_' + str(run_id) + '.csv', 'a') as out_stream:
					np.savetxt(out_stream, [np.append(np.array([individual]), individual_labels[individual])],  delimiter=", ", fmt="%s")
			break

	return run_id


NUM_NODES = 264
NUM_THREADS = 1
NUM_RUNS = 1
DATASET = 'multiple_subjects'
DATA_DIRECTORY = '../../data/' + DATASET
MAX_LABELS, MIN_LABELS = 21, 5 
SUB_LIST = []
PERCENT_THRESHOLD = 100
SAVE_DIR = './group_level_results/'

if not os.path.isdir(SAVE_DIR):
	print("Folder that will store the results cannot be found.")
	print("Creating the results folder in " + SAVE_DIR)
	os.makedirs(SAVE_DIR)

if __name__ == '__main__':
	file_list = os.listdir(DATA_DIRECTORY)
	for file in file_list:
		if file.startswith('subject'):
			SUB_LIST.append(file)

	p = Pool(NUM_THREADS)
	params = []

	for run in range(NUM_RUNS):
		params.append(tuple([run, DATA_DIRECTORY, PERCENT_THRESHOLD, SUB_LIST, MAX_LABELS , MIN_LABELS, NUM_NODES, DATASET]))

	for val in p.map(single_run, params):
		print ('Completed run: ', run)