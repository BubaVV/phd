import pandas
import statsmodels.formula.api as sm
import sys
from itertools import combinations
from multiprocessing import Pool
from scipy.stats import linregress
from sklearn import linear_model

INTERCORR_LIMIT = 0.99

def worker(a):
	global descriptors
	
	#model = linear_model.LinearRegression()
	#model.fit(descriptors[a[0]].reshape(-1,1), descriptors[a[1]])
	#about 19m user time on server
	#result = model.score(descriptors[a[0]].reshape(-1,1), descriptors[a[1]])
	#return (result, a[0], a[1])
	
	result = sm.OLS(descriptors[a[0]], descriptors[a[1]], missing='drop').fit()
	#about 20m user time on server
	return (result.rsquared, a[0], a[1])
	
	#result = linregress(descriptors[a[0]], descriptors[a[1]])
	#about 27 minutes user time on server
	#return ((result.rvalue)**2, a[0], a[1])

def worker_init(a):
	global descriptors
	descriptors = a

if __name__ == '__main__':
	descriptors = pandas.read_csv('descs.txt', delim_whitespace = True, skiprows = 2)
	descriptors = descriptors.set_index('MolID')
	descriptors.sort_index(inplace = True)
	print('File loaded for', descriptors.shape[0],'molecules and',descriptors.shape[1], 'descriptors')
	responce = pandas.read_csv('resp1.txt', delim_whitespace = True, index_col = 0, na_values = -999)
	responce.sort_index(inplace = True)
	print('Responces loaded for', responce.shape[0], 'molecules and', responce.shape[1], 'systems')

	try:
		descriptors.drop('No.', axis = 1, inplace = True)
	except ValueError:
		pass

	files_difference = set(responce.index.values).symmetric_difference(set(descriptors.index.values))
	if len(files_difference):
		print('Input files are unequal! Check these molecules:')
		for i in list(files_difference):
			print(i)
		sys.exit()
	# print(responce)
	# print(descriptors)
	# sys.exit()
	#descriptors = descriptors.iloc[:, :50]
	
	constants = []
	for i in descriptors.columns.values:
		if descriptors[i].nunique() == 1:
			constants.append(i)
	if constants != []:
		print(len(constants), 'descriptors are constants, saved to file')
		const_file = open('constants.txt','w')		
		for item in constants:
			const_file.write("%s\n" % item)
		const_file.close()
		descriptors.drop(constants, axis = 1, inplace = True)	
		descriptors.to_csv('filtered.csv', sep=';',header=True, index=True)
		print(descriptors.shape[1],'left')

	#descriptors = descriptors.loc[:,df.apply(pd.Series.nunique) != 1]

	#tasks = [descriptors[list(x)] for x in combinations(descriptors.columns.values,2)]
	tasks = combinations(descriptors.columns.values,2)
	pool = Pool(initializer=worker_init, initargs=(descriptors,))
	print('Searching descriptors cross-correlation. Can be slow')
	results = pool.imap_unordered(worker, tasks)
	pool.close()
	pool.join()
	correlated = list(filter(lambda x: x[0] > INTERCORR_LIMIT, results))

	if len(correlated) > 0:
		print(len(correlated),'correlated pairs found. Saving to file and removing from dataset')
		corr_file = open('corr_descs.txt','w')
		for i in correlated:
			corr_file.write(' '.join(map(str, i))+'\n')
		corr_file.close()
		corr_list = list(map(lambda x: x[1], correlated))
		corr_list = list(set(corr_list))
		print(len(corr_list), 'descriptors from pairs removed')
		descriptors.drop(corr_list, axis = 1, inplace = True)
		descriptors.to_csv('filtered.csv', sep=';',header=True, index=True)
		print(descriptors.shape[1],'left')
	else:
		print('No cross-correlated descriptors')
		
