import pandas
import sklearn
import sys
from sklearn import linear_model
from sklearn import cross_validation
import re

if len(sys.argv) == 1:
	print('No descriptors given, exiting')
	sys.exit()

#descriptors = pandas.read_csv('filtered.csv', delimiter = ';')
#descriptors = descriptors.set_index('MolID')
#descriptors.sort_index(inplace = True)
descriptors = pandas.read_csv('descs.txt', delim_whitespace = True, skiprows = 2)
descriptors = descriptors.set_index('MolID')
descriptors.sort_index(inplace = True)
print('File loaded for', descriptors.shape[0],'molecules and',descriptors.shape[1], 'descriptors')
responce = pandas.read_csv('resp1.txt', delim_whitespace = True, index_col = 0, na_values = -999)
responce.sort_index(inplace = True)
print('Responces loaded for', responce.shape[0], 'molecules and', responce.shape[1], 'systems:')
[print(x) for x in responce.columns.values]

files_difference = set(responce.index.values).symmetric_difference(set(descriptors.index.values))
if len(files_difference):
	print('Input files are unequal! Check these molecules:')
	for i in list(files_difference):
		print(i)
	sys.exit()

try:
	with open('forbidden.txt','r') as f:
		forbidden_templates = [x.strip() for x in f.readlines()]
		print('%d templates for forbidden molecules found' % len(forbidden_templates))
except FileNotFoundError:
	print('No forbidden dataset found')
	forbidden_templates = []
	
forbidden_mols = list(set([x for x in descriptors.index.values for y in forbidden_templates if re.match(y, x)]))
forbidden_mols.sort()
print('Forbidden dataset consists of %d molecules' % len(forbidden_mols))
descriptors.drop(forbidden_mols, inplace = True, axis = 0)
responce.drop(forbidden_mols, inplace = True, axis = 0)	
	
try:
	with open('validation.txt','r') as f:
		validation_templates = [x.strip() for x in f.readlines()]
		print('%d templates for validation molecules found' % len(validation_templates))
except FileNotFoundError:
	print('No validation dataset found')
	validation_templates = []
validation_mols = list(set([x for x in descriptors.index.values for y in validation_templates if re.match(y, x)]))
validation_mols.sort()
print('Validation dataset consists of %d molecules' % len(validation_mols))

validation_descriptors = descriptors.loc[validation_mols]
validation_responce = responce.loc[validation_mols]

descriptors.drop(validation_mols, inplace = True, axis = 0)
responce.drop(validation_mols, inplace = True, axis = 0)
print('Training dataset consists of %d molecules' % descriptors.shape[0])

#sys.exit()
	
results = sys.argv[1:]
#print(results)
#print(validation_descriptors)
#print(validation_responce)
for i in results:
	if i not in descriptors.columns.values:
		print(i, 'not in descriptors set, exiting')
		sys.exit()

for i in responce.columns.values:
	X = descriptors.ix[:,results]
	Y = responce[i]
	model = linear_model.LinearRegression()
	model.fit(X, Y)
	calculated = pandas.Series(data = model.predict(X), index = responce.index.values, name = 'Calculated')
	
	predicted = pandas.Series(data = cross_validation.cross_val_predict(model, X, Y, cv=Y.shape[0]), index = responce.index.values, name = 'Predicted')
	
	
	df = pandas.concat([X, Y, calculated, predicted], axis = 1)
	df.dropna(inplace = True)
	result_filename = ('%s_%s.csv' % (i, "_".join(results))).replace('/','_')
	df.to_csv(result_filename)
	
	#doing validation
	X = validation_descriptors.ix[:,results]
	Y = validation_responce[i]
	
	val_calculated = pandas.Series(data = model.predict(X), index = validation_responce.index.values, name = 'Calculated')
	df = pandas.concat([X, Y, val_calculated], axis = 1)
	df.dropna(inplace = True)
	result_filename = ('%s_%s_val.csv' % (i, "_".join(results))).replace('/','_')
	df.to_csv(result_filename)