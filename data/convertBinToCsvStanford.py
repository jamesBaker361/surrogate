import sys
import osmium
import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser('Bin to CSV')

inputFile_str='inputFile'
outputFile_str='outputFile'

parser.add_argument('--'+inputFile_str,'-i',nargs='?',help='input file path',default='austin.complete.gr.bin')
parser.add_argument('--'+outputFile_str,'-o',help='output file path',nargs='?')

coordinate_precision = 1000000.0

if __name__=='__main__':
    arg_vars=vars(parser.parse_args(sys.argv[1:]))
    inputFile=arg_vars[inputFile_str]
    
    with open(inputFile, 'rb') as graph_file:

        def read_int():
            return int.from_bytes(graph_file.read(4), 'little', signed=True)

        def read_string():
            buffer = []
            while True:
                i = graph_file.read(1)
                if i == b'\x00':
                    return "".join(map(chr, buffer))
                else:
                    buffer.append(int.from_bytes(i, 'big'))

        numOfVertices = read_int() 
        numOfEdges = read_int()
        print('v',numOfVertices)
        print('e',numOfEdges)
        edges=[]
        vertices=[]
        for x in range(0, numOfVertices):
            vertices.append(read_int())
        for x in range(0, numOfEdges):
            edges.append(read_int())
        numOfNodeAttributes = read_int()
        vertex_to_coordinate = {}
        atts={}
        for x in range(0, numOfNodeAttributes):
            attribute = read_string()
            size = read_int()
            #values=read_int()
            #print(read_int())
            atts[attribute]=[]
            print(attribute,size)
            
            graph_file.seek(size, os.SEEK_CUR)
        numOfEdgeAttributes=read_int()
        for y in range(0,numOfEdgeAttributes):
            attribute=read_string()
            size=read_int()
            values=read_int()
            print(attribute,size)
            atts[attribute]=[]
            for g in range(0,values):
                atts[attribute].append(read_int())
            #print(atts[attribute][:10])
            
        print('done :)')