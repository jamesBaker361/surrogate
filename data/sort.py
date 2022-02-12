import sys

def sortEdges(inputFile,outputFile):
	with open(inputFile,'r') as src:
		header=next(src).rstrip()
		edges=[]
		for line in src:
			edges.append(line.rstrip())
		edges.sort(key=lambda x: (int(x.split(',')[0]),int(x.split(',')[1])))
		edges.insert(0,header)
		with open(outputFile,'w+') as dest:
			dest.writelines([e+'\n' for e in edges])

if __name__ == '__main__':
	clargs=sys.argv[1:]
	inputFile=clargs[0]
	outputFile=clargs[1]
	sortEdges(inputFile,outputFile)