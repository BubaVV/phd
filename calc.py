import pandas
import statsmodels.formula.api as sm
import statsmodels.api as sm_api
import sys
from itertools import combinations
from multiprocessing import Pool
import json
import re

DEPTH = 4
CUT = 1000

#a = [1,2,3,4]
#b = [[1,2],[2,3],[3,4],[1,3],[2,4],[1,4]]
#d = [[x]+y for x in a for y in b if x not in y]
#d [[1, 2, 3], [1, 3, 4], [1, 2, 4], [2, 3, 4], [2, 1, 3], [2, 1, 4], [3, 1, 2], [3, 2, 4], [3, 1, 4], [4, 1, 2], [4, 2, 3], [4, 1, 3]]

def worker_init(d, r):
	global descriptors
	global responce
	descriptors = d
	responce = r
	
def worker(task):
	global descriptors
	global responce
	
	ans = 0
	x = descriptors[task]
	x = sm_api.add_constant(x)
	for i in responce.columns.values:
		result = sm.OLS(responce[i].values.reshape(-1,1), x, missing='drop').fit()
		ans += result.rsquared_adj
		#print(result.summary())
	return (ans, task)

if __name__ == '__main__':
	descriptors = pandas.read_csv('filtered.csv', delimiter = ';')
	descriptors = descriptors.set_index('MolID')
	descriptors.sort_index(inplace = True)
	print('File loaded for', descriptors.shape[0],'molecules and',descriptors.shape[1], 'descriptors')
	responce = pandas.read_csv('resp1.txt', delim_whitespace = True, index_col = 0, na_values = -999)
	responce.sort_index(inplace = True)
	print('Responces loaded for', responce.shape[0], 'molecules and', responce.shape[1], 'systems:')
	[print(x) for x in responce.columns.values]
	
	try:
		with open('forbidden.txt','r') as f:
			forbidden_templates = [x.strip() for x in f.readlines()]
			print('%d templates for forbidden molecules found' % len(forbidden_templates))
	except FileNotFoundError:
		print('No forbidden dataset found')
		forbidden_templates = []
	
	try:
		with open('validation.txt','r') as f:
			validation_templates = [x.strip() for x in f.readlines()]
			print('%d templates for validation molecules found' % len(validation_templates))
	except FileNotFoundError:
		print('No validation dataset found')
		validation_templates = []
	
	forbidden_mols = list(set([x for x in descriptors.index.values for y in forbidden_templates if re.match(y, x)]))
	forbidden_mols.sort()
	print('Forbidden dataset consists of %d molecules' % len(forbidden_mols))
	descriptors.drop(forbidden_mols, inplace = True, axis = 0)
	responce.drop(forbidden_mols, inplace = True, axis = 0)	
	validation_mols = list(set([x for x in descriptors.index.values for y in validation_templates if re.match(y, x)]))
	validation_mols.sort()
	print('Validation dataset consists of %d molecules' % len(validation_mols))
	descriptors.drop(validation_mols, inplace = True, axis = 0)
	responce.drop(validation_mols, inplace = True, axis = 0)
	print('Training dataset consists of %d molecules' % descriptors.shape[0])
	
	#print(worker(['MW','AMW','RBF']))
	#print(validation_mols, len(validation_mols))
	#sys.exit()
	#descriptors = descriptors.iloc[:, :50]
	
	for i in range(1,DEPTH+1):
		if i == 1:
			tasks = descriptors.columns.values
			tasks = list(map(lambda x: [x], tasks))
		else:
			tasks = [[x]+y for x in descriptors.columns.values for y in results if x not in y]
		#print(tasks)
		pool = Pool(initializer=worker_init, initargs=(descriptors, responce))
		print('Digging', i, 'descriptor models')
		results = pool.imap_unordered(worker, tasks)
		pool.close()
		pool.join()
		results = list(results)
		print(len(results), 'results before filtering')

		#uniqs = [frozenset(x[1]) for x in results]
		#uniqs = set(uniqs)
		
		uniqs = set()
		t = []
		for j in results:
			if frozenset(j[1]) not in uniqs:
				t.append(j)
				uniqs.add(frozenset(j[1]))
		#print(uniqs)
		print(len(uniqs), 'unique combinations')
		
		results[:] = t
		print(len(t), 'results before cut')
		
		results.sort(reverse = True)
		results = results[:CUT]
		print(results)
		with open('results_%d.json' % i,'w') as f:
			json.dump(results, f)
		results[:] = list(map(lambda x: x[1], results))
		
		