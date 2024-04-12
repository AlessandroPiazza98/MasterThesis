#SkLearn Utilities
from sklearn.metrics import ConfusionMatrixDisplay #In particular some performances utilities

#I/O Utilities
import matplotlib.pyplot as plt #For plots

def debug(dataset):
    dataset[0].validate(raise_on_error=True)
    print("dataset[0] shape: "+str(dataset[0]))
    print("Total dataset[0] nodes: "+str(dataset[0].num_nodes))
    print("Total dataset[0] edges: "+str(dataset[0].num_edges))
    print("Total dataset[0] node features: "+str(dataset[0].num_node_features))
    print("Total dataset[0] node features: "+str(dataset[0].num_edge_features))
    print("dataset[0] have isolated nodes: "+str(dataset[0].has_isolated_nodes()))
    print("dataset[0] have self loops: "+str(dataset[0].has_self_loops()))
    print("dataset[0] is a directed graph: "+str(dataset[0].is_directed()))
    print("dataset[0] class: "+str(dataset[0].y))
    print()

def save_conf_matr(Dataset, label, results_path):
    disp = ConfusionMatrixDisplay(Dataset)
    disp.plot(include_values=False, xticks_rotation=60)
    plt.title("Confusion Matrix")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    plt.savefig(results_path+"/Conf_Matr_"+label+".png", dpi=300)