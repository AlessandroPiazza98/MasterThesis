#Importing Libraries

#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch.nn import Linear as Lin
from torch.nn import ReLU
from Models import * # type: ignore

#PyTorch_Geometric Utilities
from torch_geometric.loader import  DataLoader
from Utils.class_balanced_loss import BalancedLoss

#SkLearn Utilities
from sklearn.metrics import top_k_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#Basic libraries
import numpy as np

#I/O Utilities
import matplotlib.pyplot as plt
import pickle
import argparse
import tabulate
import wandb
import os
import datetime
from itertools import groupby

os.environ["WANDB_SILENT"] = "true"


#Setup Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="Babel")
parser.add_argument("-c", "--classes", type=str, default=60)
parser.add_argument("-dt", "--data_size", type=str, default="Medium")
parser.add_argument("-dp", "--data_path", type=str, default="/data03/Users/Alessandro/Data/")
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-dv", "--device", type=str, default="auto")
parser.add_argument("-pt", "--train_test", type=float, default=0.7)
parser.add_argument("-bc", "--batch_size", type=int, default=32)
parser.add_argument('-debug', action='store_true')
parser.add_argument("-tk", "--top_k", type=int, default=5)
parser.add_argument("-rp", "--results_path", type=str, default="/home/ale_piazza/MasterThesis/Plot")
parser.add_argument("-wk", "--wandb_key", type=str, default="")
parser.add_argument('-m', "--model", type=str, default="GCN")

args = parser.parse_args()

print("Running experiments with the following configuration:")
print(tabulate.tabulate([
                ["Dataset", args.dataset], 
                ["Classes", args.classes],
                ["Size", args.data_size],
                ["Data Path",args.data_path],
                ["Epochs", args.epochs],
                ["Device", args.device],
                ["Training", str(args.train_test*100)+"%"],
                ["Batch Size", args.batch_size],
                ["Debug Data", args.debug],
                ["Top_K", args.top_k],  
                ["Results Path",args.results_path], 
                ["Wandb API",args.wandb_key!=""],
                ["Model",args.model],                               
                ], headers=['Argument', 'Value']))
print()



#Starting Graph Convolutional Network with GENConv layers

#Main
if __name__ == "__main__":
    #Set values according to Parser arguments
    classes=int(args.classes)
    train_path=args.data_path+args.dataset+str(classes)+"_train_"+args.data_size+".pkl"
    test_path=args.data_path+args.dataset+str(classes)+"_test_"+args.data_size+".pkl"
    val_path=args.data_path+args.dataset+str(classes)+"_val_"+args.data_size+".pkl"
    perc_train = args.train_test
    batch_size = args.batch_size
    num_epochs = args.epochs
    top_k = args.top_k
    results_path=args.results_path
    wandb_key = args.wandb_key
    model_key = args.model

    tz = datetime.timezone.utc
    ft = "%Y-%m-%dT%H:%M:%S%z"
    now = datetime.datetime.now(tz=tz).strftime(ft)

    #Setup WandB project

    if wandb_key != "":
        wandb.login(key=args.wandb_key)
        wandb.init(
            # set the wandb project where this run will be logged
            project="Master Thesis",
            name="GCN_"+args.dataset+str(classes)+"_"+args.data_size+"_"+now,
            # track hyperparameters and run metadata
            config={
            "architecture": "GCN",
            "dataset": args.dataset+str(classes)+"_"+args.data_size,
            "epochs": num_epochs,
            }
        )

    #Use CUDA if available, otherwise selected device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Experiments will run using "+str(device)+" on "+str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device(args.device)
        if args.device == "cpu":
            print("Experiments will run using "+str(device))
        else:
            print("Experiments will run using "+str(device)+" on "+str(torch.cuda.get_device_name(device)))

    #Load Training Set from pickle file
    print("\nLoading dataset from "+train_path)
    with open(train_path, 'rb') as file:
            train_dataset = pickle.load(file)
    print("Dataset "+args.dataset+str(classes)+"_"+args.data_size+" is ready\n")

    #Load Test Set from pickle file
    print("\nLoading dataset from "+test_path)
    with open(test_path, 'rb') as file:
            test_dataset = pickle.load(file)
    print("Dataset "+args.dataset+str(classes)+"_"+args.data_size+" is ready\n")

    #Load Validation Set from pickle file
    print("\nLoading dataset from "+val_path)
    with open(val_path, 'rb') as file:
            val_dataset = pickle.load(file)
    print("Dataset "+args.dataset+str(classes)+"_"+args.data_size+" is ready\n")

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

    if args.debug:

        "Information about first observation of the training set:"
        debug(train_dataset)

        "Information about first observation of the test set:"
        debug(test_dataset)

        "Information about first observation of the validation set:"
        debug(val_dataset)


    #Split the data
    N_th_train_test = int(len(train_dataset)*perc_train)

    #TODO keep attention val/test names and definitions
    train_loader = DataLoader(train_dataset[:N_th_train_test], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_dataset[N_th_train_test:], batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, )

    
    input_dim = train_dataset[0].num_node_features  # Number of features for each node (name, x, y, z)
    edge_dim = train_dataset[0].num_edge_features
    output_dim = classes # Number of classes (number of unique tasks) #TODO 
    labels = list(range(0,output_dim))
    class_freq = []

    for i in range(0, len(train_dataset)):
         class_freq.append(train_dataset[i].y)
    class_freq = [len(list(group)) for key, group in groupby(sorted(class_freq))]
    class_w = torch.Tensor([x/sum(class_freq) for x in class_freq]).to(device)

    # Define your model and optimizer
    MODELS_MAP = models_map(input_dim, output_dim, edge_dim, device) # type: ignore
    model = MODELS_MAP[model_key]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)

    if args.data_size =="Full":
        criterion = torch.nn.CrossEntropyLoss(class_w)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("Start debug")
    if args.debug:
        model_debug = MODELS_MAP[model_key]
        optimizer = torch.optim.Adam(model_debug.parameters(), lr=0.008)
        model_debug.train()
        data = next(iter(train_loader))
        print(data) # Iterate in batches over the training dataset.  
        optimizer.zero_grad()  # Clear gradients.
        data = data.to(device)
        out = model_debug(data.x, data.edge_index, data.edge_attr, data.batch, debug=args.debug)
        print(args.debug) # Perform a single forward pass.
        #loss_func = BalancedLoss(samples_per_cls=class_freq, no_of_classes=classes,
        #                     loss_type='cross-entropy', beta=0.99, gamma=0.5, device=device, label_smoothing=0.0) #TODO Debug dimensions
        #print(out.shape)
        #print(data.y.shape)
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        torch.cuda.empty_cache()
    print("End debug")


    print("Model Configuration:")
    print(model)
    print("\nWill run for "+str(num_epochs)+" epochs with batch size equal to "+str(batch_size))
    print()


    def train():
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.  
            
            optimizer.zero_grad()  # Clear gradients.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch) # Perform a single forward pass.
            #loss_func = BalancedLoss(samples_per_cls=class_freq, no_of_classes=classes,
            #                     loss_type='cross-entropy', beta=0.99, gamma=0.5, device=device, label_smoothing=0.0) #TODO Debug dimensions
            #print(out.shape)
            #print(data.y.shape)
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            torch.cuda.empty_cache()

    def test(loader, labels,k):
        model.eval()
        correct = 0
        loss_ = 0
        top_k = 0
        conf_matr = np.zeros((len(labels), len(labels)))
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)  
            loss = criterion(out, data.y)
            loss_ += loss.item()
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())
            torch.cuda.empty_cache()
            top_k += top_k_accuracy_score(data.y.cpu().numpy(), out.detach().cpu().numpy(), labels=labels, normalize=False, k=k)
            conf_matr += confusion_matrix(data.y.cpu().numpy(),pred.cpu().numpy(), labels=labels)
            
             

        return correct / len(loader.dataset), loss_ / len(loader.dataset), top_k / len(loader.dataset), conf_matr  # Derive ratio of correct predictions.
   
    # Training loop
    train_acc_val = []
    val_acc_val = []
    train_loss_val = []
    val_loss_val = []
    train_top_k_val = []
    val_top_k_val = []
    for epoch in range(1,num_epochs+1):
        train()
        train_acc, train_loss, train_top_k, train_conf_matr = test(train_loader, labels, top_k)
        val_acc, val_loss, val_top_k, val_conf_matr = test(val_loader, labels, top_k)
        train_acc_val.append(train_acc)
        val_acc_val.append(val_acc)

        train_loss_val.append(train_loss)
        val_loss_val.append(val_loss)

        train_top_k_val.append(train_top_k)
        val_top_k_val.append(val_top_k)

        torch.cuda.empty_cache()

        print(tabulate.tabulate([
                ["Training Set", f'{train_acc:.4f}', f'{train_loss:.4f}',f'{train_top_k:.4f}'], 
                ["Validation Set", f'{val_acc:.4f}', f'{val_loss:.4f}',f'{val_top_k:.4f}']
                ], headers=[f'Epoch: {epoch:03d}', "Accuracy", "Loss","Top_"+str(top_k)+" Accuracy"],))
        print()


        if wandb_key != "":
            # log metrics to wandb #TODO
            wandb.log({"train_acc": train_acc, 
                    "train_loss": train_loss, 
                    "train_top_"+str(top_k):train_top_k,
                    "val_acc": val_acc, 
                    "val_loss": val_loss, 
                    "val_top_"+str(top_k):val_top_k
                    })
            
    train_acc, train_loss, train_top_k, train_conf_matr = test(train_loader, labels, top_k)
    val_acc, val_loss, val_top_k, val_conf_matr = test(val_loader, labels, top_k)
    test_acc, test_loss, test_top_k, test_conf_matr = test(test_loader, labels, top_k)

    disp = ConfusionMatrixDisplay(train_conf_matr)
    disp.plot(include_values=False, xticks_rotation=60)
    plt.title("Confusion Matrix")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    plt.savefig("MasterThesis/Plot/Conf_Matr_Train.png", dpi=300)


    disp = ConfusionMatrixDisplay(val_conf_matr)
    disp.plot(include_values=False, xticks_rotation=60)
    plt.title("Confusion Matrix")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    plt.savefig("MasterThesis/Plot/Conf_Matr_Val.png", dpi=300)

    disp = ConfusionMatrixDisplay(test_conf_matr)
    disp.plot(include_values=False, xticks_rotation=60)
    plt.title("Confusion Matrix")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    plt.savefig("MasterThesis/Plot/Conf_Matr_Test.png", dpi=300)

    if wandb_key != "":
            # log metrics to wandb #TODO
            wandb.log({"test_acc": test_acc, 
                    "test_loss": test_loss, 
                    "test_top_"+str(top_k):test_top_k,
                    })
    if wandb_key != "":  
        wandb.finish()
