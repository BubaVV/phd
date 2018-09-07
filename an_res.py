import json
import os
import sys
import pandas

BIN_SIZE = 10

counter = 1
while True:
	if not os.path.isfile('results_%d.json' % counter):
		print('No results_%d.json, exiting' % counter)
		sys.exit()
	print('results_%d.json present' % counter)
	with open('results_%d.json' % counter, 'r') as f:
		j = json.load(f)
	print(len(j), 'results')
	j.sort(reverse = True)
	#print(j)
	uniq_descs = []
	for i in j:
		uniq_descs.extend(i[1])
	uniq_descs[:] = list(set(uniq_descs))
	#print(uniq_descs)
	df = pandas.DataFrame(index = uniq_descs, columns = range(0,len(j),BIN_SIZE), data = 0)
	for i in range(0,len(j),BIN_SIZE):
		t = j[i:i+BIN_SIZE]
		#print(i)
		bin_descs = []
		[bin_descs.extend(x[1]) for x in t]
		#print(bin_descs)
		
		for desc in uniq_descs:
			df.ix[desc, i] = bin_descs.count(desc)
	#df['Sum'] = df.sum(axis = 1)
	#df.sort_values('Sum', axis = 0, ascending = False, inplace = True)
	df.sort_values(list(df.columns.values), inplace = True, ascending = False)
	df.to_csv('desc_freqs_%d.csv' % counter)
	counter += 1