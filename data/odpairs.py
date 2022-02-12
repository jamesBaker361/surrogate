import sys
import osmium
import os
import numpy as np
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser(description='tntp -> csv')

pyhissim_str='pyhissim'
pbf_str='pbf'
graph_str='graph'
event_str='events'

parser.add_argument('--'+pyhissim_str,'-p',help='pyhissim file',nargs='?', default='physsim-network.xml',type=str)
parser.add_argument('--'+pbf_str,'-b',help='pbf input',nargs='?', default='texas-austin.osm.pbf')
parser.add_argument('--'+graph_str,'-g',help='binary graph input file',nargs='?', default='austin.gr.bin')
parser.add_argument('--'+event_str,'-e',help='events file',nargs='?',default='0.events.csv')

coordinate_precision = 100000.0

#event_tree = ET.parse('/home/rajnikant/Desktop/beam/output/sf-light/sf-light-1k-xml__2022-02-10_17-28-41_jgx/ITERS/it.0/0.events.xml')

#pyhssim_tree = ET.parse('/home/rajnikant/Desktop/beam/test/input/sf-light/r5/physsim-network.xml')

#pbf = '/home/rajnikant/Desktop/beam/test/input/sf-light/r5/sflight_muni.osm.pbf'

#graph_file = '/home/rajnikant/Desktop/routing-framework/data/sflight/graph.gr.bin'

class OSMHandler(osmium.SimpleHandler):

    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.id_to_node_ids = dict()
        self.id_to_node_coordinate = dict()

    def node(self, n):
        self.id_to_node_coordinate[n.id] = n.location ##str(n.location.lat)+"/"+str(n.location.lon)

    def way(self, w):
        self.id_to_node_ids[w.id] =  [int(str(node)) for node in w.nodes]

    def get_coordinates_for_way_id(self, first_way_id):
        node_ids = self.id_to_node_ids.get(first_way_id)
        if node_ids != None:
            return [self.id_to_node_coordinate.get(node_id) for node_id in node_ids]
if __name__=='__main__':
    arg_vars=vars(parser.parse_args(sys.argv[1:]))
    
    pbf=arg_vars[pbf_str]
    pyhssim_tree=ET.parse(arg_vars[pyhissim_str])
    graph_file=arg_vars[graph_str]
    event_tree = ET.parse(arg_vars[event_str])
    
    osm_handler = OSMHandler()
    osm_handler.apply_file(pbf)

    with open(graph_file, 'rb') as graph_file:

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
        for x in range(0, numOfVertices):
            read_int()
        for x in range(0, numOfEdges):
            read_int()
        numOfAttributes = read_int()
        vertex_to_coordinate = {}
        for x in range(0, numOfAttributes):
            attribute = read_string()
            size = read_int()
            print(attribute)
            if attribute == 'lat_lng':
                read_int()
                for i in range(0 , numOfVertices):
                    vertex_to_coordinate[i] = (read_int() / coordinate_precision, read_int()/ coordinate_precision) 
            else:
                graph_file.seek(size, os.SEEK_CUR)

        coord_to_vertex = {}
        pts = []
        for x in range(0, numOfVertices):
            v_to_c = vertex_to_coordinate[x]
            coord_to_vertex[v_to_c] = x
            pts.append([v_to_c[0], v_to_c[1]])

        kdtree = KDTree(np.array(pts),leafsize=2)

    physsim_root = pyhssim_tree.getroot()
    link_id_with_orig_id = {}
    for links in physsim_root:
        if links.tag == 'links':
            for link in links:
                id = link.attrib.get('id')
                for attributes in link:
                    for attribute in attributes:
                        if attribute.attrib.get('name') == 'origid':
                            link_id_with_orig_id[id] = int(attribute.text)


    event_root = event_tree.getroot()
    od_pairs_file = open(r"beam-od-pair.csv","w+")
    od_pairs_file.write("origin,destination\n")
    #print(coord_to_vertex)
    for child in event_root:
        attribute = child.attrib
        if attribute.get('type') == 'PathTraversal' and len(attribute.get('links')) > 0:
            links = attribute.get('links').split(",")
            first_link_id = links[0]
            last_link_id = links[-1]
            first_way_id = link_id_with_orig_id.get(first_link_id)
            last_way_id = link_id_with_orig_id.get(last_link_id)
            origin_list = osm_handler.get_coordinates_for_way_id(first_way_id)
            destination_list = osm_handler.get_coordinates_for_way_id(last_way_id)
            origin = origin_list[0] if origin_list != None else None
            destination = destination_list[-1] if destination_list != None else None
            if destination != None and origin != None and origin !=  destination:
                origin_point = (origin.lat, origin.lon)
                destination_point = (destination.lat, destination.lon)
                oid = coord_to_vertex.get(origin_point, kdtree.query_ball_point((origin.lat, origin.lon), r=0.01)[0])
                did = coord_to_vertex.get(destination_point, kdtree.query_ball_point((destination.lat, destination.lon), r=0.01)[0])
                od_pairs_file.write(str(oid)+","+str(did)+"\n")


    od_pairs_file.close()