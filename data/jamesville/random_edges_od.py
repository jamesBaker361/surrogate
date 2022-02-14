import random
import networkx as nx

def all_nodes_connected(graph):
    n=graph.number_of_nodes()
    for i in range(n):
        for j in range(i+1,n,1):
            if nx.has_path(graph,i,j)==False:
                return False
    return True

def make_random_graph(n=100,f=0.1,limit=1000):
    prune=int(n*n*f)
    g=nx.complete_graph(n)
    removed=0
    for l in range(limit):
        pair=random.sample(range(n), 2)
        i=pair[0]
        j=pair[1]
        if g.has_edge(i,j):
            g.remove_edge(i,j)
            if all_nodes_connected(g)==True:
                removed+=1
                if removed>=prune:
                    return g
            else:
                g.add_edge(i,j)
    return g

def graph_to_csv(g,filename):
    with open(filename,'w+') as f:
        f.write("edge_tail,edge_head,length,capacity,speed")
        for (t,h) in sorted([e for e in g.edges()]):
            length=random.randint(10, 20)
            capacity=random.randint(500, 1500)
            speed=random.randint(30, 80)
            f.write('\n{},{},{},{},{}'.format(t,h,length,capacity,speed))
            
def demand_csv(n,filename,limit=5):
    demand=set()
    with open(filename,'w+') as f:
        f.write("origin,destination,volume")
        while len(demand)<limit:
            [i,j]=random.sample(range(n), 2)
            volume=random.randint(10, 50)
            if (i,j) not in demand:
                demand.add((i,j))
                f.write('\n{},{},{}'.format(i,j,volume))
            
if __name__=='__main__':
    for n in [10,20,30]:
        for f in [0.0,0.1,0.5,0.9]:
            g=make_random_graph(n=n,f=0)
            graph_to_csv(g,'edges-{}-{}.csv'.format(n,f))
            demand_csv(n,'demand-{}-{}.csv'.format(n,f))