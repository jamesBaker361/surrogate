import os
import random
scratch=os.environ["SCRATCH"]

#this randomly augments edges by turning their capacity to 1

def perturb(edge_file,label_file,perturbed_file,frac=.4):
    """
    perturbs (100*frac)% of the edges by setting capacity to 1
    
    Parameters
    ----------
    edge_file: str
        path to edge file for reading
    label_file: str
        path to binary label file for writing
    perturbed_file: str
        path to perturbed edges file for writing
    frac: float
        proportion of edges to perturb
        
    Returns
    -------
    None
    """
    with open(edge_file,'r') as src, open(label_file,'w') as label_dest, open(perturbed_file,'w') as dest:
        label_dest.write('tail,head,label')
        dest.write(next(src).strip())
        for r in src:
            row=r.strip().split(',')
            [edge_tail,edge_head,length,capacity,speed]=row
            label=1
            if random.random()<=frac:
                label=0
                capacity=str(100)
            label_dest.write('\n{},{},{}'.format(edge_tail,edge_head,label))
            dest.write('\n'+','.join([edge_tail,edge_head,length,capacity,speed]))
            
def perturbDiscrete(edge_file,label_file,perturbed_file,frac=.1,unit=100):
    """
    perturbs (100*frac)% of the edges by +/- (1,2)*unit
    
    Parameters
    ----------
    edge_file: str
        path to edge file for reading
    label_file: str
        path to binary label file for writing
    perturbed_file: str
        path to perturbed edges file for writing
    frac: float
        proportion of edges to perturb
    unit: int
        discrete amount to perturb capacity by
        
    Returns
    -------
    None
    """
    with open(edge_file,'r') as src, open(label_file,'w') as label_dest, open(perturbed_file,'w') as dest:
        dest.write(next(src).strip())
        label_dest.write('tail,head,label')
        for r in src:
            row=r.strip().split(',')
            [edge_tail,edge_head,length,capacity,speed]=row
            label=1
            if random.random()<=frac:
                label=0 #perturbed!
                capacity=str(int(capacity)+random.choice([-2*unit,1*unit,unit,2*unit]))
            label_dest.write('\n{},{},{}'.format(edge_tail,edge_head,label))
            dest.write('\n'+','.join([edge_tail,edge_head,length,capacity,speed]))
                
        
    
if __name__ =='__main__':
    #perturb('data/jamesburg/edges.csv','data/jamesburg/labels.csv','data/jamesburg/edgesMixed.csv')
    #perturb(scratch+'/data/AustinShrunk/edges.csv',scratch+'/data/AustinShrunkPerturbed/labels.csv',scratch+'/data/AustinShrunkPerturbed/edgesMixed.csv')
    edge_file=scratch+'/data/AustinShrunk/edges.csv'
    perturbed_file=scratch+'/data/AustinShrunk/perturbedEdges.csv'
    label_file=scratch+'/data/AustinShrunk/labels.csv'
    perturb(edge_file,label_file,perturbed_file)