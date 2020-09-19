# imports for CDM function
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
from ipywidgets import interact, fixed, IntSlider, FloatSlider, SelectionSlider, Dropdown
import itertools
from collections import Counter

class bcolors:
	""" Custom class for storing colours of warnings, errors, etc """
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'

# Conditional Dependence Monitor Main Code
def CDGAM_fast(L_dev, k = 2, sig = 0.01, policy = 'new', verbose = False, return_more_info = False):
	# create pd dataframe
	df = pd.DataFrame(data=L_dev, columns=["LF_"+str(i) for i in range(L_dev.shape[1])])
	no_other_LFs = L_dev.shape[1]-2
	def create_CT_tables(df, L_dev, k):
		"""create all combinations of contingency table's of {LF_i, LF_j, [LF_all others]}"""

		CT_list = []
		for i in range(L_dev.shape[1]):
			for j in [k2 for k2 in range(i, L_dev.shape[1]) if k2!=i]:
				other_LFs_list = [df['LF_'+str(m)] for m in range(L_dev.shape[1]) if (m!=i and m!=j)]
				CT = pd.crosstab(other_LFs_list + [df['LF_'+str(i)]], df['LF_'+str(j)], margins = False) 

				# prep to reindex only the LF column closest to values (this is new in the fast version)
				indices_other_LFs = [] # first get current indices of other LF columns
				for itera in range(len(CT.index)):
					indices_other_LFs.append(CT.index[itera][:-1])
				indices_other_LFs = list(set(indices_other_LFs)) # get unique values only
				indices_closest_LF = [(i-1,) for i in range(k+1)]
				all_indices = [ ind1+ind2 for ind1 in indices_other_LFs for ind2 in indices_closest_LF]
				k_list = [i-1 for i in range(k+1)]
				#CT = CT.reindex(index=list(itertools.product(*[tuple(k_list)] * (no_other_LFs+1))), columns=k_list, fill_value=0)
				# reindex only the LF column closest to values
				CT = CT.reindex(index=all_indices, columns=k_list, fill_value=0)
				CT_list.append(CT)
		return CT_list
	
	def show_CT(q, CT_list):
		"""function to return qth CT at index q-1 of CT_list"""
		return CT_list[q-1]

	# create and show all CT's (if verbose) with slider applet
	CT_list = create_CT_tables(df, L_dev, k) # create all combinations of Contingency Tables
	if verbose:
		print("No of tables = Choosing 2 from "+str(L_dev.shape[1])+" LFs = "+str(L_dev.shape[1])+"C2 = "+str(len(CT_list)))
		#print("Note: Showing subtables where CI is not clearly evident")
		interact(show_CT, q=IntSlider(min=1, max=len(CT_list), value=0, step=1), CT_list=fixed(CT_list))

	def get_conditional_deps(CT_list, sig, k, delta = 0):
		"""peform 3-way table chi-square independence test and obtain test statistic chi_square 
		(or corresponding p-value) for each CT table where each CT-table is a ({k+1}^{no of other LFs})x(k+1)x(k+1) matrix
		https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html"""
		CD_edges = []; CD_nodes = []; CD_edges_p_vals = []; p_vals_sum_dict = {}; CT_reduced_list = []
		count = 0; #n_bad = 0
		for CT in CT_list:
			count+=1; Z = int(len(CT.values)/3) # no of rows/3 (this is new in the fast version)
			CT_reshaped = np.reshape(CT.values, (Z,k+1,k+1)) 

			if policy == 'old':
				p_val_tot, CT_reduced = get_p_total_old_policy(CT_reshaped, k, Z, delta, sig, count, verbose, return_more_info) # Older policy of one round of 0 row/col reduction and then adding delta
			else:
				p_val_tot, CT_reduced = get_p_total_new_policy(CT_reshaped, k, Z, sig, count, verbose, return_more_info) # newer policy of complete reduction
			CT_reduced_list.append(CT_reduced)
			# checking if total p_value is lesser than chosen sig
			if p_val_tot < sig: 
				digits_LF1 = CT.index.names[no_other_LFs][3:]
				digits_LF2 = CT.columns.name[3:] # 3rd index onwards to remove LF_ (3 characters)
				CD_edges.append( (int(digits_LF1), int(digits_LF2)) )
				CD_edges_p_vals.append(p_val_tot)

		#print info
		edges_info_dict = {}
		edges_info_dict['CD_edges'] = CD_edges; edges_info_dict['CD_edges_p_vals'] = CD_edges_p_vals
		if verbose:
			edges_df = pd.DataFrame(edges_info_dict)
			print(edges_df)

		return edges_info_dict, CT_reduced_list

	# retrieve modified CTs & tuples of LFs that are conditionally independent
	edges_info_dict, CT_reduced_list = get_conditional_deps(CT_list, sig = sig, k = k, delta = 1)

	# display reduced contingency tables' submatrices whose all elements = delta is not true
	if verbose:
		non_delta_tuple_indices = []
		for q in range(len(CT_reduced_list)):
			for r in range(len(CT_reduced_list[q])):
				delta = 1
				if ~(CT_reduced_list[q][r]==delta).all():
					non_delta_tuple_indices.append( ((q,r), (q,r)) ) # apending a tuple of tuples because first part of outer tuple is key and second is value passed gy slider to fn

		def show_CT_sub_matrices(t, CT_reduced_list):
			"""function to return qth CT at index q-1 of CT_list"""
			return CT_reduced_list[t[0]][t[1]]
		print("The reduced and modified contingency tables with non-delta values are given below")
		interact(show_CT_sub_matrices, t=Dropdown(description='index tuples (table number, submatrix number)', options=non_delta_tuple_indices), CT_reduced_list=fixed(CT_reduced_list))
	
	if return_more_info:
		return edges_info_dict
	else:
		return edges_info_dict['CD_edges']

##################################################################
# Heuristic Policies to reduce Contingency Tables and get P values
##################################################################
def get_p_total_old_policy(CT, k, Z, delta, sig, count, verbose, return_more_info):
	def zero_row(m):
		return np.any(np.all(m == 0, axis=1) == True)
	def zero_col(m):
		return np.any(np.all(m == 0, axis=0) == True)
	def remove_first_0_row_column(m):
		first_0_col_index = list(np.all(m == 0, axis=0)).index(True)
		first_0_row_index = list(np.all(m == 0, axis=1)).index(True)
		rowmask = np.ones(len(m), dtype=bool)
		colmask = np.ones(len(m), dtype=bool)
		colmask[first_0_col_index] = False
		rowmask[first_0_row_index] = False
		m1 = m[:,colmask]
		m2 = m1[rowmask]
		return m2
	chi2_sum = 0; dof_sum = 0; M_reduced = []
	for i in range(Z):
		m = CT[i]
		if k==2:
			if zero_row(m) and zero_col(m): # if a zero row and zero col are there, remove it to get 2x2 matrix
				m_new = remove_first_0_row_column(m)
				if zero_row(m_new) or zero_col(m_new): # if still there is a zero row or column, add delta
					m_new = m_new + delta
			else: # if combo of zero row and column is not present
				if zero_row(m) or zero_col(m): # but if there is either a zero row or a zero column
					m_new = m + delta
				else:
					m_new = m
		elif k==3:
			if zero_row(m) and zero_col(m): # if a zero row and zero col are there, remove it to get 3x3 matrix
				m_temp = remove_first_0_row_column(m)
				if zero_row(m_temp) and zero_col(m_temp): # if still a zero row and zero col are there, remove it to get 2x2 matrix
					m_new = remove_first_0_row_column(m_temp)
					if zero_row(m_new) or zero_col(m_new): # lastly if there is a zero row or column, add delta
						m_new = m_new + delta
				else:
					if zero_row(m_temp) or zero_col(m_temp): # if a second combo of zero row and column wasn't there but one of the two is, add delta
						m_new = m_temp + delta
					else:
						m_new = m_temp
			else: # if combo of zero row and column is not present
				if zero_row(m) or zero_col(m): # but if there is either a zero row or a zero column
					m_new = m + delta
				else:
					m_new = m
		else:
			print("only k = cardinality = 2 or 3 can be handled!")
			#m_new = m
		M_reduced.append(m_new)

	for i in range(Z):
		m_new = M_reduced[i]
		if ~(np.all(m_new==delta)): # if not (all elements of 2x2 or 3x3 matrix are 0+delta), then
			n_rows = m_new.shape[0]; n_cols = m_new.shape[1]
			chi2stat, p, dof, exp_freq = chi2_contingency(m_new)
			if p<sig: # if any submatrix is dependent, whole CT is dependent
					if verbose: print("left at submatrix ", i," of CT ", count)
					return p, M_reduced
			chi2_sum += chi2stat
			dof_sum += (n_rows-1)*(n_cols-1)
	
	p_val_tot = 1-chi2.cdf(chi2_sum, dof_sum)
	return p_val_tot, M_reduced

def get_p_total_new_policy(M, k, Z, sig, count, verbose, return_more_info):
	def is0D(m):
		return m.shape[0] == 0 or m.shape[1] == 0
	def is1D(m):
		return m.shape[0] == 1 or m.shape[1] == 1
	def remove_all_0_rows_cols(m):
		# remove all 0 columns
		m2 = m[:, ~np.all(m == 0, axis=0)]
		# remove all 0 rows
		m3 = m2[~np.all(m2 == 0, axis=1)] 
		return m3
	#if verbose: print(count)
	chi2_sum = 0
	dof_sum = 0; no_1D = 0; no_0D = 0; M_reduced = []
	for i in range(Z):
		m_new = remove_all_0_rows_cols(M[i])
		M_reduced.append(m_new)
		#if verbose: print(m_new)

	for i in range(Z):
		m_new = M_reduced[i]
		if is0D(m_new):
			no_0D += 1
			#if verbose: print("skipping 0d")
			continue
		elif is1D(m_new):
			no_1D += 1 # assume all 1D reduced matrices are conditionally independent
			#if verbose: print("skipping 1d")
			continue
		else: # square and not square matrices (~2x3 or 3x2)
			n_rows = m_new.shape[0]
			n_cols = m_new.shape[1]

			chi2stat, p, dof, exp_freq = chi2_contingency(m_new)
			if p<sig: # if any submatrix is dependent, whole CT is dependent
				if verbose: print("left at submatrix ", i," of CT ", count)
				return p, M_reduced
			chi2_sum += chi2stat
			dof_sum += (n_rows-1)*(n_cols-1)
	if no_0D+no_1D != Z: # if all reduced matrices are not 1d, 0d
		p_val_tot = 1-chi2.cdf(chi2_sum, dof_sum)
	else: 
		p_val_tot = 1 # to be considered independent, set any value > alpha here
	return p_val_tot, M_reduced