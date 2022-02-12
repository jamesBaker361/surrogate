import argparse
import sys

parser = argparse.ArgumentParser(description='tntp -> csv')

function_str='function'
inputFile_str='inputFile'
outputFile_str='outputFile'

parser.add_argument('--'+function_str,'-f',help='network or demand')
parser.add_argument('--'+inputFile_str,'-i',help='input file path')
parser.add_argument('--'+outputFile_str,'-o',help='output file path')

def readOD(inputFile):
    od = {}
    origin_no = 0  # our starting Origin number in case the file doesn't begin with one
    with open(inputFile, "r") as f:
        for line in f:
            line = line.rstrip()  # we're not interested in the newline at the end
            if not line:  # empty line, skip
                continue
            if line.startswith("Origin"):
                origin_no = int(line[7:].strip())  # grab the integer following Origin
            else:
                elements = line.split(";")  # get our elements by splitting by semi-colon
                if len(elements)==1:
                    continue
                for element in elements:  # loop through each of them:
                    if not element:  # we're not interested in the last element
                        continue
                    try:
                        element_no, element_value = element.split(":")  # get our pair
                    except ValueError:
                        print(elements)
                    # beware, these two are now most likely padded strings!
                    # that's why we'll strip them from whitespace and convert to integer/float
                    if(origin_no != int(element_no.strip())):
                        od[(origin_no, int(element_no.strip()))] = float(element_value.strip())
    return od


def readNetwork(inputFile):
	#input= init_node1	term_node2	capacity3	length4	free_flow_time5	b6	power7	speed8	toll	link_type
	#output="edge_tail,edge_head,length,capacity,speed"
    network = {}
    with open(inputFile, "r") as f:
        line=next(f).rstrip()
        while line!='<END OF METADATA>':
            line=next(f).rstrip()
        for line in f:
            line = line.rstrip()
            line = line.split(";")[0].split('\t')
            if len(line)==1 or line[0]=='~':
                continue
            #print([line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8]])
            network[line[1], line[2]] = [line[1], line[2], line[4], line[3], line[8]]
    return network






def printOD_flows(outFile,od):
    with open(outFile, "w") as f:
        tmpOut = "origin,destination,volume"
        f.write(tmpOut+"\n")
        for d in od:
            tmpOut = str(d[0])+","+str(d[1])+","+str(od[d])
            f.write(tmpOut+"\n")

def printNetwork(outFile,network):

    with open(outFile, "w") as f:
        tmpOut = "edge_tail,edge_head,length,capacity,speed"
        f.write(tmpOut+"\n")
        for link in network:
            tmpOut =','.join(network[link])
            f.write(tmpOut+"\n")

if __name__ == '__main__':
	arg_vars=vars(parser.parse_args(sys.argv[1:]))
	function=arg_vars[function_str]
	inputFile=arg_vars[inputFile_str]
	outputFile=arg_vars[outputFile_str]
	if function=='network':
		network=readNetwork(inputFile)
		printNetwork(outputFile,network)
	else:
		od=readOD(inputFile)
		printOD_flows(outputFile,od)