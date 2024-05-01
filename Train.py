#Importing Libraries

#PyTorch and Models.py
import torch
from Models import * # type: ignore

#PyTorch_Geometric Utilities
from torch_geometric.loader import  DataLoader # torch_geometric DataLoader
from Utils.class_balanced_loss import BalancedLoss #TODO actually not working

#SkLearn Utilities
from sklearn.metrics import top_k_accuracy_score, confusion_matrix #In particular some performances utilities

#Basic libraries and utilities
import numpy as np
from Utils.Utils import * #Import all functions from Utils.py

#I/O Utilities
import pickle #To handle the pickle file in input with the Data
import argparse #To parse arguments
import tabulate #To print well-formatted tables
import wandb #To export results on wandb platform through an API key
import os #System 
import datetime #TODO To keep track of computational times
from itertools import groupby #Utilities to make group opeations

os.environ["WANDB_SILENT"] = "true" #Don't want the classical verbose output of wandb


#Setup Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="Babel") #The dataset considered for the computations Babel/NTU
parser.add_argument("-c", "--classes", type=str, default=120) #The number of classes to consider 60/120
parser.add_argument("-dt", "--data_size", type=str, default="Full") #The dataset size to consider Small/Medium/Full
parser.add_argument("-dp", "--data_path", type=str, default="/data03/Users/Alessandro/Data/") #Path to the folder with the pickle files
parser.add_argument("-e", "--epochs", type=int, default=100) #Number of epochs
parser.add_argument("-dv", "--device", type=str, default="cuda:1") #Type of device auto/cpu/cuda
parser.add_argument("-pt", "--train_test", type=float, default=0.7) #TODO Split of training and Test samples (must be removed) 
parser.add_argument("-bc", "--batch_size", type=int, default=32) #Batchsize considered (high vakues can give memory issue)
parser.add_argument('-debug', action='store_true') #Activate debug mode that prints some useful information
parser.add_argument("-tk", "--top_k", type=int, default=5) #Number of K considered for the computation of top K accuracy
parser.add_argument("-rp", "--results_path", type=str, default="/home/ale_piazza/MasterThesis/Plot") #Path in which plots are exported
parser.add_argument("-wk", "--wandb_key", type=str, default="") #API key for wandb. If not inserted nothing is exported.
parser.add_argument('-m', "--model", type=str, default="GCN") #Model to consider: the string must match with one of the keys of Models.models_map
parser.add_argument('-l', "--loss", type=str, default="CE") #Loss function
parser.add_argument('-lr', "--learn_rate", type=float, default=0.001) #Learning rate
parser.add_argument('-hn', "--hidden", type=int, default=64) #Hidden layers if requested

args = parser.parse_args() #Initialize parser

print("Running experiments with the following configuration:") #Print table with the input arguments
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
                ["Hidden",args.hidden],
                ["Loss",args.loss],   
                ["Learning Rate",args.learn_rate],                           
                ], headers=['Argument', 'Value']))
print()

#Main
if __name__ == "__main__":

    #Set values according to Parser arguments
    classes=int(args.classes)

    #Paths to training/test/validation sets
    train_path=args.data_path+args.dataset+str(classes)+"_train_"+args.data_size+".pkl" 
    test_path=args.data_path+args.dataset+str(classes)+"_test_"+args.data_size+".pkl"
    val_path=args.data_path+args.dataset+str(classes)+"_val_"+args.data_size+".pkl"

    perc_train = args.train_test #TODO to remove after replacing the validation/test data considered
    batch_size = args.batch_size
    num_epochs = args.epochs
    top_k = args.top_k
    results_path=args.results_path
    wandb_key = args.wandb_key
    model_key = args.model
    loss_key = args.loss
    lr = args.learn_rate
    hidden = args.hidden

    tz = datetime.timezone.utc
    ft = "%Y-%m-%dT%H:%M:%S%z"
    now = datetime.datetime.now(tz=tz).strftime(ft) #Create datetime reference for the start of the computation


    #Use CUDA if available, otherwise selected device and print where the experiments will run
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

    #If in debug mode, print the shape and the cardinalities of the first graph of each dataset 
    if args.debug:
        print("\n##### DEBUG MODE ON #####\n")
        "Information about first observation of the training set:"
        debug(train_dataset)
        "Information about first observation of the test set:"
        debug(test_dataset)
        "Information about first observation of the validation set:"
        debug(val_dataset)

    #Split the data TODO behaviour to remove to consider the direct dataset samples
    N_th_train_test = int(len(train_dataset)*perc_train)

    #TODO keep also attention val/test names and definitions, actually perc_train% of train_dataset is the Training, the reamining % is the Validation and val_dataset becomes the Test
    train_loader = DataLoader(train_dataset[:N_th_train_test], batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(train_dataset[N_th_train_test:], batch_size=batch_size, shuffle=False)

    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #Dimensions considered by the model
    input_dim = train_dataset[0].num_node_features #Number of features for each node
    edge_dim = train_dataset[0].num_edge_features #Number of features for each edge
    output_dim = classes #Number of classes 
    labels = list(range(0,output_dim)) #Assigning a numerical label for each class
    class_freq = []
    class_w = []
    #Compute the weight that must be considered by the loss function for each class
    for i in range(0, len(train_dataset[:N_th_train_test])):
         class_freq.append(train_dataset[i].y)
         if torch.isnan(train_dataset[i].x).any():
              raise ValueError(i)
    #Compute how frequent is each class
    if loss_key == "CEw":
        class_f = [len(list(group)) for key, group in groupby(sorted(class_freq))]
        class_w = torch.Tensor([1/x for x in class_f]).to(device)
        print(class_w)
        #If in debug mode, print the dicionaris of frequencies and weights
        if args.debug:
                class_keys = [key.numpy()[0] for key, group in groupby(sorted(class_freq))]
                freq_dict = dict(zip(class_keys, class_f))
                weight_dict = dict(zip(class_keys, class_w))
                print("Dictionary of class frequencies in Training Set:")
                print(freq_dict)
                print("\nDictionary of class weights in Training Set:")
                print(weight_dict)
    #Define your model and optimizer, taking the model from the models_map dictionary on Models.py 
    MODELS_MAP = models_map(input_dim, output_dim, edge_dim, device, hidden) # type: ignore
    model = MODELS_MAP[model_key]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001) #TODO go in deeper details for optimizer
    #Select different criterions only if size is Full to avoid mismatch in the number of labels
    if args.data_size == "Full":
        criterion = select_loss(loss_key, class_w)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    #Print the size of the data after each layer in the network, making a batch to pass through a model identical to the one that will be trained
    if args.debug:
        model_debug = MODELS_MAP[model_key]
        model_debug.train()
        data = next(iter(train_loader)) #Pick a single batch
        print("\nThe debugging batch has this shape:")
        print(data) # Print size of the batch  
        print("\nAfter each layer this is the expected size of the data:")
        optimizer.zero_grad()  # Clear gradients.
        data = data.to(device)
        out = model_debug(data.x, data.edge_index, data.edge_attr, data.batch, debug=args.debug) #Run model on debug mode
        print("\n##### DEBUG MODE OFF #####\n")
        #loss_func = BalancedLoss(samples_per_cls=class_freq, no_of_classes=classes,
        #                     loss_type='cross-entropy', beta=0.99, gamma=0.5, device=device, label_smoothing=0.0) #TODO Debug dimensions

    #Print model configuration and number of epochs/batch_size
    print("Model Configuration:")
    print(model)
    print("\nWill run for "+str(num_epochs)+" epochs with batch size equal to "+str(batch_size))
    print()

    #Setup WandB project
    if wandb_key != "":
        wandb.login(key=args.wandb_key) #Login on wnadb thorugh API key parsed
        wandb.init(
            #Set the wandb project where this run will be logged
            project="Master Thesis",
            name=model_key+"_"+args.dataset+str(classes)+"_"+args.data_size+"_"+now, #The name depends on the model/classes/size considered and uses the datetime as key
            #Track hyperparameters and run metadata
            config={
            "architecture": model_key,
            "dataset": args.dataset+str(classes)+"_"+args.data_size,
            "epochs": num_epochs,
            "batch-size": batch_size,
            "loss": loss_key,
            "learn_rate": lr,
            "layers": model,
            "hidden": hidden
            }
        )

    #Training function
    def train():
        model.train()
        for data in train_loader:  #Iterate in batches over the training dataset.  
            optimizer.zero_grad()  #Clear gradients.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch) #Perform a single forward pass obtaining network output
            #loss_func = BalancedLoss(samples_per_cls=class_freq, no_of_classes=classes,
            #                     loss_type='cross-entropy', beta=0.99, gamma=0.5, device=device, label_smoothing=0.0) #TODO Debug dimensions
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  #Derive gradients.
            optimizer.step()  #Update parameters based on gradients.

    #Test function
    def test(loader, labels,k):
        model.eval()
        #Initialize indexes and confusion matrix
        correct = 0
        loss_ = 0
        top_k = 0
        conf_matr = np.zeros((len(labels), len(labels)))
        acc_by_label = np.zeros(len(labels))
        count_by_label = np.zeros(len(labels))
        for data in loader:  #Iterate in batches over the  dataset
            data = data.to(device) #Move data on device
            out = model(data.x, data.edge_index, data.edge_attr, data.batch) #Obtain predictions 
            loss = criterion(out, data.y) 
            loss_ += loss.item() #Update loss
            pred = out.argmax(dim=1)  #Use the class with highest value as predicitons
            #Update indexes and confusion matrix values
            correct += int((pred == data.y).sum())
            top_k += top_k_accuracy_score(data.y.cpu().numpy(), out.detach().cpu().numpy(), labels=labels, normalize=False, k=k)
            conf_matr += confusion_matrix(data.y.cpu().numpy(),pred.cpu().numpy(), labels=labels)
              
            count_by_label += count_labels(data.y.cpu().numpy(), labels)
            acc_by_label += count_matching_labels(data.y.cpu().numpy(), pred, labels)

        acc_by_label = np.divide(acc_by_label, count_by_label, out=np.zeros_like(acc_by_label), where=count_by_label!=0)
        return correct / len(loader.dataset), loss_ / len(loader.dataset), top_k / len(loader.dataset), conf_matr, sum(acc_by_label)/len(acc_by_label)  # Derive ratio of correct predictions.
   
    #Training loop
    #Initialize arrays for performances throuhg the epochs
    train_acc_val = []
    val_acc_val = []
    train_loss_val = []
    val_loss_val = []
    train_top_k_val = []
    val_top_k_val = []
    #STart iterations thourgh epochs
    for epoch in range(1,num_epochs+1):
        #Train the model
        train()

        #Test the model on training and validation set
        train_acc, train_loss, train_top_k, train_conf_matr, train_top_1_norm = test(train_loader, labels, top_k)
        val_acc, val_loss, val_top_k, val_conf_matr, val_top_1_norm = test(val_loader, labels, top_k)

        if wandb_key != "":
            # log metrics to wandb #TODO
            wandb.log({"train_acc": train_acc, 
                    "train_loss": train_loss, 
                    "train_top_"+str(top_k):train_top_k,
                    "train_top_1_norm":train_top_1_norm,
                    "val_acc": val_acc, 
                    "val_loss": val_loss, 
                    "val_top_"+str(top_k):val_top_k,
                    "val_top_1_norm":val_top_1_norm
                    })
        

        #Print a summary tables of the epoch
        print(tabulate.tabulate([
                ["Training Set", f'{train_acc:.4f}', f'{train_loss:.4f}',f'{train_top_k:.4f}',f'{train_top_1_norm:.4f}'], 
                ["Validation Set", f'{val_acc:.4f}', f'{val_loss:.4f}',f'{val_top_k:.4f}',f'{val_top_1_norm:.4f}']
                ], headers=[f'Epoch: {epoch:03d}', "Accuracy", "Loss","Top_"+str(top_k)+" Accuracy", "Top-1 Norm"],))
        print()
    
    #Obtain all the predictions after last epoch
    test_acc, test_loss, test_top_k, test_conf_matr, test_top_1_norm = test(test_loader, labels, top_k)

    #Generates and save the confusion matrix for all the dataset
    

    #Export results on wandb
    if wandb_key != "":
        # log metrics to wandb #TODO
        save_conf_matr(train_conf_matr, args.dataset, classes, "Train", results_path, now)
        save_conf_matr(val_conf_matr, args.dataset, classes, "Val", results_path, now)
        save_conf_matr(test_conf_matr, args.dataset, classes, "Test", results_path, now)
        wandb.log({
                "test_acc": test_acc, 
                "test_loss": test_loss, 
                "test_top_"+str(top_k):test_top_k,
                "test_top_1_norm":test_top_1_norm
                })
    #Close wandb session
    if wandb_key != "":  
        wandb.finish()
