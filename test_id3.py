
import id3

def print_tree_structure_util (node, indent):
    for i in range(indent):
        print(' ',end='')
    if node.is_leaf():
        print(' <',node.conclusion,'>')
    else:
        print('split',node.splitFeature)
        for V in node.child.keys():
            for i in range(indent):
                print(' ',end='')
            print('case',V,end='')
            print_tree_structure_util(node.child[V],indent+1)

def print_tree_structure (id3):
    root=id3.root
    print_tree_structure_util (root, 0)

def read_examples (file, positive='True'):
    f=open(file,'r')
    examples=[]
    for line in f.readlines():
        line=line.rstrip().strip()
        ex=line.split('\t')
        n=len(ex)
        if ex[n-1]==positive:
            ex[n-1]=True
        else:
            ex[n-1]=False
        examples.append(ex)
    f.close()
    return examples

if __name__ == "__main__":
    ex1=read_examples('test1.txt','Win')
    t1=id3.id3().train(ex1, True)
    #print_tree_structure(t1)
    print(t1.classify_label(['Morning', 'Grand Slam','Grass','Yes'] \
        ,'Win','Lose','Inconclusive')) #Win
    print(t1.classify_label(['Afternoon', 'Friendly','Clay','No'] \
        ,'Win','Lose','Inconclusive'))   #Lose

    t1_2=id3.id3().train(ex1, False) # No default value
    #print_tree_structure(t1_2)
    print(t1_2.classify_label(['Afternoon', 'Friendly','Clay','No'] \
        ,'Win','Lose','Inconclusive'))   #Inconclusive
    
    ex2=read_examples('test2.txt','Infected')
    t2_1=id3.id3().train(ex2,True)
    print_tree_structure(t2_1)
    pass