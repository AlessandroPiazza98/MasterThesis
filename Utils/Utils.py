#Torch
import torch
from torch_geometric.data import Data

#SkLearn Utilities
from sklearn.metrics import ConfusionMatrixDisplay #In particular some performances utilities

#I/O Utilities
import matplotlib.pyplot as plt #For plots
import wandb
import matplotlib.pylab as pylab


params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large'}
pylab.rcParams.update(params)

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

def save_conf_matr(Dataset, dataset, classes, label, results_path, now):
    name = "Conf_Matr_"+dataset+str(classes)+label+"_"+now
    title = "Confusion Matrix "+dataset+"_"+str(classes)+"_"+label
    disp = ConfusionMatrixDisplay(Dataset)
    disp.plot(include_values=False, xticks_rotation=60)
    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    plt.savefig(results_path+"/"+name+".png", dpi=300)
    plt.savefig(results_path+"/"+name+".eps", dpi=300, format="eps")

    wandb.log({name: plt})

def select_loss(loss_key, class_w):
        if loss_key == "CEw":
            return torch.nn.CrossEntropyLoss(class_w)
        elif loss_key == "CE":
            return torch.nn.CrossEntropyLoss()
        else:
            return torch.nn.CrossEntropyLoss()
        
def count_labels(arr, total_labels):
    label_counts = {}
    for label in arr:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    label_occurrences = [label_counts.get(i, 0) for i in total_labels]
    return label_occurrences

def count_matching_labels(arr1, arr2, total_labels):
    label_counts = {}
    for label1, label2 in zip(arr1, arr2):
        if label1 == label2:
            if label1 in label_counts:
                label_counts[label1] += 1
            else:
                label_counts[label1] = 1
    
    label_occurrences = [label_counts.get(i, 0) for i in total_labels]
    return label_occurrences

def win_split(data, a, b, device): #TODO
    kpts=25
    x = data.x[a*kpts:(b+1)*kpts,:]

    edge_index = data.edge_index
    mask = (edge_index[0] >= a*kpts) & (edge_index[0] <= (b+1)*kpts-1) & (edge_index[1] >= (a)*kpts) & (edge_index[1] <= (b+1)*kpts-1)
    edge_index = torch.stack([edge_index[0][mask],edge_index[1][mask]])
    edge_index = torch.sub(edge_index, a*kpts)

    edge_attr = data.edge_attr[mask]
    y = data.y

    data_out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=torch.zeros(x.shape[0], dtype=torch.int64).to(device))
    return data_out

def windowing(data, k, device, overlap=0): #TODO
    kpts = 25
    windowed_data = []
    frames = int((len(data.x)/kpts)/k)
    for i in range(k):
        if i ==0:
            a = frames*i
        else:
            a = frames*i-int(overlap/2)
        if i == k-1:
            b = frames*(i+1)
        else:
            b = frames*(i+1)+int(overlap/2)
        windowed_data.append(win_split(data, a, b, device))
    return windowed_data
