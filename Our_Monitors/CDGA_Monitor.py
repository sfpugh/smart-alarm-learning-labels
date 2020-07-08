# imports for CDM function
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
from ipywidgets import interact, fixed, IntSlider, FloatSlider
import itertools

# Conditional Dependence Monitor Main Code
def CDGAM(L_dev, k = 2, sig = 0.01, verbose = False):
	# create pd dataframe
	df = pd.DataFrame(data=L_dev, columns=["LF_"+str(i) for i in range(L_dev.shape[1])])
	no_other_LFs = L_dev.shape[1]-2
	def create_CT_tables(df, L_dev, k):
		"""create all combinations of contingency table's of {LF_i, LF_j, [LF_all others]}"""
		CT_list = []
		for i in range(L_dev.shape[1]):
			for j in [k for k in range(i, L_dev.shape[1]) if k!=i]:
				other_LFs_list = [df['LF_'+str(m)] for m in range(L_dev.shape[1]) if (m!=i and m!=j)]
				CT = pd.crosstab(other_LFs_list + [df['LF_'+str(i)]], df['LF_'+str(j)], margins = False) 
				CT = CT.reindex(index=list(itertools.product(*[(-1, 0, 1)] * (no_other_LFs+1))), columns=[-1,0,1], fill_value=0)
				CT_list.append(CT)
		return CT_list

	def show_CT(q, CT_list):
		"""function to return qth CT at index q-1 of CT_list"""
		return CT_list[q-1]

	# create and show all CT's with slider applet
	CT_list = create_CT_tables(df, L_dev, k) # create all combinations of Contingency Tables
	if verbose:
		print("No of tables = Choosing 2 from "+str(L_dev.shape[1])+" LFs = "+str(L_dev.shape[1])+"C2 = "+str(len(CT_list)))
		print("Note: Showing subtables where CI is not clearly evident")
	interact(show_CT, q=IntSlider(min=1, max=len(CT_list), value=0, step=1), CT_list=fixed(CT_list));

	class bcolors:
		""" Custom class for storing colours of warnings, errors, etc """
		HEADER = '\033[95m'
		OKBLUE = '\033[94m'
		OKGREEN = '\033[92m'
		WARNING = '\033[93m'
		FAIL = '\033[91m'
		ENDC = '\033[0m'

	def get_p_vals_tables(CT_list, sig, k, delta = 0):
		"""peform 3-way table chi-square independence test and obtain test statistic chi_square 
		(or corresponding p-value) for each CT table where each CT-table is a ({k+1}^{no of other LFs})x(k+1)x(k+1) matrix
		https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html"""
		p_val_list = []
		LFs_that_have_deps = []
		count = 0; n_bad = 0; CT_list_2 = []
		for CT in CT_list:
			count+=1; Z = (k+1)**no_other_LFs
			CT_reshaped = np.reshape(CT.values, (Z,k+1,k+1)) 

			# check for "both 0 row and column" in (k+1)x(k+1) matrices; reduce to from Zx(k+1)x(k+1) to Zx(k)x(k) matrix
			zero_col_counter = np.zeros(k+1); zero_row_counter = np.zeros(k+1); 
			CT_reshaped_2 = np.zeros((Z,k,k))
			for i in range(Z): 
				zero_col_counter += (CT_reshaped[i,:,:]==0).all(axis=0) # find bools representing 0 columns/vertical dirn (axis = 0); add bool result to counter
				zero_row_counter += (CT_reshaped[i,:,:]==0).all(axis=1) # similarly for row
			if (zero_col_counter==0).all() == False and (zero_row_counter==0).all() == False:
				zero_col_index = np.where(zero_col_counter == Z)[0][0] # get index of *first* 0 col in all (k+1)x(k+1) matrices
				zero_row_index = np.where(zero_row_counter == Z)[0][0] # similarly for row
				for i in range(Z): 
					temp = np.delete(CT_reshaped[i,:,:], zero_col_index, axis=1)
					temp2 = np.delete(temp, zero_row_index, axis=0)
					CT_reshaped_2[i,:,:] = temp2

			# check for any zero columns / rows in both (k)x(k) matrices in CT; if yes, add delta to all values
			bad_table = False
			for i,j in [(i,j) for i in range(Z) for j in [0,1]]: 
				if ~np.all(CT_reshaped_2[i,:,:].any(axis=j)):
					bad_table = True
					n_bad += 1
			if bad_table:
				if delta!=0:
					# to prevent 0 row/col in exp_freq table which in turn prevents division by 0 in statistic
					CT_reshaped_2 = CT_reshaped_2 + delta
					if verbose:
						print("Adding delta to table ", count)
				else:
					if verbose:
						print(bcolors.WARNING + "Error : table ",count," has a zero column/row in one (or both) of its 2x2 matrices!" + bcolors.ENDC)
					continue
			
			# calculate statistic for each (k)x(k) matrix and sum
			chi2_sum = 0; actual_Z = 0
			for i in range(Z):
				if ~(CT_reshaped_2[i,:,:]==delta).all(): # if not (all elements of kxk matrix are 0+delta), then
					chi2stat, p, dof, exp_freq = chi2_contingency(CT_reshaped_2[i,:,:])
					chi2_sum += chi2stat; actual_Z += 1
			if verbose: print("There are ",Z-actual_Z,"/",Z," ",k,"x",k," zero matrices that weren't used in chi^2 computation")
			p_val_tot = 1-chi2.cdf(chi2_sum, actual_Z*(k-1)*(k-1))
			
			p_val_list.append(p_val_tot)
			# checking if total p_value is lesser than chosen sig
			if p_val_tot < sig: 
				if verbose:
					print("table: {0:<15} chi-sq {1:<15} p-value: {2:<15} ==> ~({3} __|__ {4} | LF_all others)".format(count, np.around(chi2_sum,4), np.around(p_val_tot,6), str(CT.index.names[no_other_LFs]), str(CT.columns.name)))
				if len(CT.index.names[1])==5:
					digits_LF1 = CT.index.names[1][-2:]
				else:
					digits_LF1 = CT.index.names[1][-1:]
				if len(CT.columns.name)==5:
					digits_LF2 = CT.columns.name[-2:]
				else:
					digits_LF2 = CT.columns.name[-1:]
				LFs_that_have_deps.append( (int(digits_LF1), int(digits_LF2)) )
			else:
				if verbose:
					print("table: {0:<15} chi-sq {1:<15} p-value: {2:<15}".format(count, np.around(chi2_sum,4), np.around(p_val_tot,6)))
			CT_list_2.append(CT_reshaped_2.astype(int))
		print("\nDependecy Graph Edges: ", LFs_that_have_deps)
		
		if n_bad!=0 and delta == 0 and verbose:
			print(bcolors.OKBLUE+"\nNote"+bcolors.ENDC+": Either tune delta (currently "+str(delta)+") or increase datapoints in dev set to resolve"+bcolors.WARNING+" Errors"+bcolors.ENDC)
		
		return LFs_that_have_deps, CT_list_2

	# retrieve modified CTs & tuples of LFs that are conditionally independent
	LFs_that_have_deps, CT_list_2 = get_p_vals_tables(CT_list, sig = sig, k = k, delta = 1)
	if verbose:
		print("The reduced and modified contingency tables are given below. Scroll to view all")
		interact(show_CT, q=IntSlider(min=1, max=len(CT_list), value=0, step=1), CT_list=fixed(CT_list_2));
	
	return LFs_that_have_deps

