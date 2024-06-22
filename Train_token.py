#Importing Libraries

#PyTorch and Models.py
import torch
from Models import * # type: ignore
from torch.nn.utils.rnn import pad_sequence

#PyTorch_Geometric Utilities
from torch_geometric.loader import  DataLoader # torch_geometric DataLoader
from torch_geometric.data import Batch
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

#TODO Comment the whole code and make more solid and compact, then add export of all information and models (also classifier). Then big tests...

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
parser.add_argument("-tk", "--top_k", type=int, default=5) #Number of K considered for the computation of top K accuracy
parser.add_argument("-rp", "--results_path", type=str, default="/home/ale_piazza/MasterThesis/Results") #Path in which plots are exported
parser.add_argument("-wk", "--wandb_key", type=str, default="") #API key for wandb. If not inserted nothing is exported.
parser.add_argument('-m', "--model", type=str, default="GCN") #Model to consider: the string must match with one of the keys of Models.models_map
parser.add_argument('-l', "--loss", type=str, default="CE") #Loss function
parser.add_argument('-lr', "--learn_rate", type=float, default=0.001) #Learning rate
parser.add_argument('-hn', "--hidden", type=int, default=64) #Hidden layers if requested
parser.add_argument('-dr', "--dropout", type=float, default=0.1) #Dropout considered on GCN2 and GCNSAG models
parser.add_argument('-re', "--regularization", type=float, default=0.000000001) #Weight decay for regularization
parser.add_argument('-nt', "--ntoken", type=int, default=10) #Hidden layers if requested
parser.add_argument('-to', "--token_overlap", type=int, default=4) #Hidden layers if requested
parser.add_argument('-learn_features', action='store_true') #Activate debug mode that prints some useful information
parser.add_argument('-id', "--model_id", type=str) #Model to consider: the string must match with one of the keys of Models.models_map
parser.add_argument('-cl', "--classifier", type=str, default="ClassifierWin") #Model to consider: the string must match with one of the keys of Models.models_map
parser.add_argument('-nh', "--nhead", type=int, default=8) #Model to consider: the string must match with one of the keys of Models.models_map
parser.add_argument('-en', "--encoder_layers", type=int, default=6) #Model to consider: the string must match with one of the keys of Models.models_map
parser.add_argument('-ff', "--feedforward", type=int, default=2048) #Model to consider: the string must match with one of the keys of Models.models_map
parser.add_argument('-learn_tokens', action='store_true') #Activate debug mode that prints some useful information
parser.add_argument('-pat', "--patience", type=int, default=3) #Model to consider: the string must match with one of the keys of Models.models_map

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
                ["Top_K", args.top_k],  
                ["Results Path",args.results_path], 
                ["Wandb API",args.wandb_key!=""],
                ["Model",args.model], 
                ["Hidden",args.hidden],
                ["Loss",args.loss],   
                ["Learning Rate",args.learn_rate],
                ["Dropout",args.dropout],  
                ["Regularization",args.regularization], 
                ["Token per graph",args.ntoken],  
                ["Token overlap",args.token_overlap],  
                ["Learn Features", args.learn_features],     
                ["Model ID", args.model_id], 
                ["Classifier", args.classifier],
                ["N Head", args.nhead],
                ["Encoder Layers", args.encoder_layers],
                ["Dim FeedForward", args.feedforward],
                ["Learn by Tokens", args.learn_tokens],
                ["Early stopping patience", args.patience]
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
    dropout = args.dropout
    reg = args.regularization
    ntoken = args.ntoken
    token_overlap = args.token_overlap
    learn_features = args.learn_features
    model_id = args.model_id
    typeClassifier = args.classifier
    nhead = args.nhead
    num_encoder_layers = args.encoder_layers
    dim_feedforward = args.feedforward
    learn_tokens = args.learn_tokens
    patience = args.patience


    tz = datetime.timezone.utc
    ft = "%Y-%m-%dT%H:%M:%S%z"
    now = datetime.datetime.now(tz=tz).strftime(ft) #Create datetime reference for the start of the computation



    #Use CUDA if available, otherwise selected device and print where the experiments will run
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Experiments will run using "+str(device)+" on "+str(torch.cuda.get_device_name(device))) 
    else:
        if args.device == "cpu":
            device = torch.device("cpu")
            print("Experiments will run using "+str(device))
        else:
            device = torch.device("cuda")
            os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev = args.device.split(':')[1]
            print("Experiments will run using "+str(device)+" on "+str(torch.cuda.get_device_name(device)))

    #Load Training Set from pickle file
    print("\nLoading training dataset from "+train_path)
    with open(train_path, 'rb') as file:
            train_dataset = pickle.load(file)
    print("Dataset "+args.dataset+str(classes)+"_"+args.data_size+" is ready\n")

    #Load Validation Set from pickle file
    print("\nLoading validation dataset from "+val_path)
    with open(val_path, 'rb') as file:
            val_dataset = pickle.load(file)
    print("Dataset "+args.dataset+str(classes)+"_"+args.data_size+" is ready\n")


    #Dimensions considered by the model
    input_dim = train_dataset[0].num_node_features #Number of features for each node
    edge_dim = train_dataset[0].num_edge_features #Number of features for each edge
    output_dim = classes #Number of classes 
    labels = list(range(0,output_dim)) #Assigning a numerical label for each class
    
    #Define your model and optimizer, taking the model from the models_map dictionary on Models.py 
    MODELS_MAP = model_feat_map(input_dim, output_dim, edge_dim, device, hidden, dropout) # type: ignore
    CLASSIFIER_MAP = classifier_map(output_dim, device, hidden, dropout, ntoken, nhead, num_encoder_layers, dim_feedforward)
      

    if learn_features and model_key == "GCN":
        model = MODELS_MAP[model_key]
        classifier = CLASSIFIER_MAP["Classifier"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg) #TODO go in deeper details for optimizer
        criterion = torch.nn.CrossEntropyLoss()
        print("Model Configuration:")
        print(model)
    elif learn_features and model_key == "GAE": #TODO include better
        encoder = GraphEncoder(input_dim, hidden, edge_dim, hidden, dropout).to(device)
        decoder = GraphDecoder(hidden, hidden, input_dim, edge_dim, dropout).to(device)
        model = GraphAutoencoder(encoder, decoder).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        criterion = nn.MSELoss()
        print("Model Configuration:")
        print(model)
    else:
        if model_key == "GAE":
            encoder = GraphEncoder(input_dim, hidden, edge_dim, hidden, dropout).to(device)
            decoder = GraphDecoder(hidden, hidden, input_dim, edge_dim, dropout).to(device)
            model = GraphAutoencoder(encoder, decoder).to(device)
        else:
            model = MODELS_MAP[model_key]
        classifier = CLASSIFIER_MAP[typeClassifier]
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=reg)
        criterion = torch.nn.CrossEntropyLoss()
        print("Classifier Configuration:")
        print(classifier)

    learn_train_dataset = []
    learn_val_dataset = []
    for i in range(int(len(train_dataset))):
        graph = train_dataset[i]
        train_dataset[i].windowed_data = windowing(graph, ntoken, device, token_overlap)  
        if learn_features and learn_tokens:
            for tkn in range(ntoken):
                learn_train_dataset.append(train_dataset[i].windowed_data[tkn])
    for i in range(int(len(val_dataset))):
        graph = val_dataset[i]
        val_dataset[i].windowed_data = windowing(graph, ntoken, device, token_overlap) 
        if learn_features and learn_tokens:
            for tkn in range(ntoken): 
                learn_val_dataset.append(val_dataset[i].windowed_data[tkn]) 

    if learn_features and learn_tokens:
        train_dataset = learn_train_dataset
        val_dataset = learn_val_dataset
#Split the training data in training/validation 
    N_th_train_test = int(len(train_dataset)*perc_train)

    #perc_train% of train_dataset is the Training, the remaining % is the Validation and val_dataset becomes the Test
    train_loader = DataLoader(train_dataset[:N_th_train_test], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_dataset[N_th_train_test:], batch_size=batch_size, shuffle=False)

    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)    

    print("\nWill run for "+str(num_epochs)+" epochs with batch size equal to "+str(batch_size))
    print()

    if learn_tokens:
        representationType = "Tokens"
    else:
        representationType = "Full"

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
            "hidden": hidden,
            "dropout" : dropout,
            "regularization" : reg,
            "representation" : model_key,
            "representationType" : representationType,
            "classifier" : typeClassifier,
            "patience" : patience
            }
        )

    if learn_features and model_key == "GCN":
            #Training function
        def train():
            model.train()
            classifier.train()
            for data in train_loader:  #Iterate in batches over the training dataset.  
                optimizer.zero_grad()  #Clear gradients.
                data = data.to(device)
                features = model(data, data.batch) #Perform a single forward pass obtaining network output
                out = classifier(features)
                loss = criterion(out, data.y)  # Compute the loss.
                loss.backward()  #Derive gradients.
                optimizer.step()  #Update parameters based on gradients.

        #Test function
        def test(loader, labels,k):
            model.eval()
            classifier.eval()
            #Initialize indexes and confusion matrix
            correct = 0
            loss_ = 0
            top_k = 0
            conf_matr = np.zeros((len(labels), len(labels)))
            acc_by_label = np.zeros(len(labels))
            count_by_label = np.zeros(len(labels))
            for data in loader:  #Iterate in batches over the  dataset
                data = data.to(device) #Move data on device
                features = model(data, data.batch) #Obtain predictions 
                out = classifier(features) #Obtain predictions 
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
    
    elif learn_features and model_key == "GAE":
        #Training function
        def train():
            model.train()
            for data in train_loader:  #Iterate in batches over the training dataset.  
                optimizer.zero_grad()  #Clear gradients.
                data = data.to(device)
                features = model(data) #Perform a single forward pass obtaining network output
                loss = criterion(features, data.x)  # Compute the loss.
                loss.backward()  #Derive gradients.
                optimizer.step()  #Update parameters based on gradients.

        #Test function
        def test(loader, labels,k):
            model.eval()
            #Initialize indexes and confusion matrix
            loss_ = 0
            for data in loader:  #Iterate in batches over the  dataset
                data = data.to(device) #Move data on device
                features = model(data) #Obtain predictions 
                loss = criterion(features, data.x) 
                loss_ += loss.item() #Update loss
            return loss_ / len(loader.dataset) # Derive ratio of correct predictions.
    
    else:
        #Training funtion
        model.load_state_dict(torch.load(results_path+"/Model"+model_id+".pt"))
        def train():
            #TODO handle import of the Model
            model.eval()
            classifier.train()
            
            for data in train_loader:  #Iterate in batches over the training dataset.
                optimizer.zero_grad()  #Clear gradients.    
                batched_tokens=torch.empty((0)).to(device)
                data = data.to(device)
                y = data.y.to(device)
                batched_tokens = []
                lengths = []
                for graph in data.to_data_list():
                    tokens = []
                    #windowed_data = windowing(graph, ntoken, device, token_overlap)
                    if model_key == "GAE":
                        for win_graph in graph.windowed_data:
                            representation = model.encoder(win_graph).to(device)
                            tokens.append(representation)
                    else:
                        for win_graph in graph.windowed_data:
                            representation = model(win_graph, win_graph.batch)
                            tokens.append(representation)
                    if typeClassifier !="Transformer":
                        combined_tokens = torch.cat(tokens, dim=1)
                        batched_tokens = torch.cat([torch.Tensor(batched_tokens).to(device), combined_tokens], dim=0)
                    else:
                        combined_tokens = torch.cat(tokens, dim=0)
                        lengths.append(combined_tokens.size(0))
                        batched_tokens.append(combined_tokens)
                if typeClassifier == "Transformer":
                    # Pad the sequences
                    padded_tokens = pad_sequence(batched_tokens, batch_first=True).to(device)

                    # Ensure the input to the transformer is of shape (batch_size, sequence_length, embedding_size)
                    padded_tokens = padded_tokens.unsqueeze(0) if padded_tokens.dim() == 2 else padded_tokens
                    # Create mask
                    max_length = max(lengths)
                    mask = torch.zeros((len(batched_tokens), max_length), dtype=torch.bool).to(device)
                    for i, length in enumerate(lengths):
                        mask[i, :length] = 1
                    out = classifier(padded_tokens)
                else:
                    out = classifier(batched_tokens)
                

                loss = criterion(out, y) # Compute the loss.
                loss.backward() #Derive gradients.
                optimizer.step()  #Update parameters based on gradients.
                
        #Test function
        def test(loader, labels,k):
            model.eval()
            classifier.eval()
            #Initialize indexes and confusion matrix
            correct = 0
            loss_ = 0
            top_k = 0
            conf_matr = np.zeros((len(labels), len(labels)))
            acc_by_label = np.zeros(len(labels))
            count_by_label = np.zeros(len(labels))
            with torch.no_grad():
                for data in loader:  #Iterate in batches over the  dataset
                    data = data.to(device)
                    y = data.y
                    batched_tokens = []
                    lengths = []
                    for graph in data.to_data_list():
                        tokens = []
                        #windowed_data = windowing(graph, ntoken, token_overlap)
                        if model_key == "GAE":
                            for win_graph in graph.windowed_data:
                                representation = model.encoder(win_graph).to(device)
                                tokens.append(representation)
                        else:
                            for win_graph in graph.windowed_data:
                                representation = model(win_graph, win_graph.batch)
                                tokens.append(representation) #Perform a single forward pass obtaining network output
                        if typeClassifier !="Transformer":
                            combined_tokens = torch.cat(tokens, dim=1)
                            batched_tokens = torch.cat([torch.Tensor(batched_tokens).to(device), combined_tokens], dim=0)
                        else:
                            combined_tokens = torch.cat(tokens, dim=0)
                            lengths.append(combined_tokens.size(0))
                            batched_tokens.append(combined_tokens)
                    if typeClassifier == "Transformer":
                        # Pad the sequences
                        padded_tokens = pad_sequence(batched_tokens, batch_first=True).to(device)

                        # Ensure the input to the transformer is of shape (batch_size, sequence_length, embedding_size)
                        padded_tokens = padded_tokens.unsqueeze(0) if padded_tokens.dim() == 2 else padded_tokens
                        # Create mask
                        max_length = max(lengths)
                        mask = torch.zeros((len(batched_tokens), max_length), dtype=torch.bool).to(device)
                        for i, length in enumerate(lengths):
                            mask[i, :length] = 1
                        out = classifier(padded_tokens)
                    else:
                        out = classifier(batched_tokens)
                    loss = criterion(out, y) 
                    loss_ += loss.item() #Update loss
                    pred = out.argmax(dim=1)  #Use the class with highest value as predicitons
                    #Update indexes and confusion matrix values
                    correct += int((pred == y).sum())
                    top_k += top_k_accuracy_score(y.cpu().numpy(), out.detach().cpu().numpy(), labels=labels, normalize=False, k=k)
                    conf_matr += confusion_matrix(y.cpu().numpy(),pred.cpu().numpy(), labels=labels)
                    
                    count_by_label += count_labels(y.cpu().numpy(), labels)
                    acc_by_label += count_matching_labels(y.cpu().numpy(), pred, labels)

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
    best_val_loss = float('inf')
    epochs_no_improvement = 0

    for epoch in range(1,num_epochs+1):
        
        #Train the model
        train()
        if learn_features and model_key == "GCN":
            #Test the model on training and validation set
            train_acc, train_loss, train_top_k, train_conf_matr, train_top_1_norm = test(train_loader, labels, top_k)
            val_acc, val_loss, val_top_k, val_conf_matr, val_top_1_norm = test(val_loader, labels, top_k)
            #Print a summary tables of the epoch
            print(tabulate.tabulate([
                    ["Training Set", f'{train_acc:.4f}', f'{train_loss:.4f}',f'{train_top_k:.4f}',f'{train_top_1_norm:.4f}'], 
                    ["Validation Set", f'{val_acc:.4f}', f'{val_loss:.4f}',f'{val_top_k:.4f}',f'{val_top_1_norm:.4f}']
                    ], headers=[f'Epoch: {epoch:03d}', "Accuracy", "Loss","Top_"+str(top_k)+" Accuracy", "Top-1 Norm"],))
            print()
        elif learn_features and model_key == "GAE":
            #Test the model on training and validation set
            train_loss = test(train_loader, labels, top_k)
            val_loss = test(val_loader, labels, top_k)
            print(tabulate.tabulate([
                    ["Training Set", f'{train_loss:.4f}'], 
                    ["Validation Set", f'{val_loss:.4f}']
                    ], headers=[f'Epoch: {epoch:03d}', "Loss"],))
            print()
            if wandb_key != "":
                # log metrics to wandb #TODO
                wandb.log({
                        "train_loss": train_loss,  
                        "val_loss": val_loss
                        })
        else:
            #Test the model on training and validation set
            train_acc, train_loss, train_top_k, train_conf_matr, train_top_1_norm = test(train_loader, labels, top_k)
            val_acc, val_loss, val_top_k, val_conf_matr, val_top_1_norm = test(val_loader, labels, top_k)
            #Print a summary tables of the epoch
            print(tabulate.tabulate([
                    ["Training Set", f'{train_acc:.4f}', f'{train_loss:.4f}',f'{train_top_k:.4f}',f'{train_top_1_norm:.4f}'], 
                    ["Validation Set", f'{val_acc:.4f}', f'{val_loss:.4f}',f'{val_top_k:.4f}',f'{val_top_1_norm:.4f}']
                    ], headers=[f'Epoch: {epoch:03d}', "Accuracy", "Loss","Top_"+str(top_k)+" Accuracy", "Top-1 Norm"],))
            print()

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
            # Check if validation loss has improved
        if val_loss < best_val_loss-0.0001:
            if learn_features:
                torch.save(model.state_dict(), results_path+"/Model"+model_id+".pt")
            best_val_loss = val_loss
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            
        # Check for early stopping
        if epochs_no_improvement >= patience:
            print(f"No improvement for {patience} epochs. Early stopping...")
            break
        
        if epoch == num_epochs:
            torch.save(model.state_dict(), results_path+"/Model"+model_id+".pt")
    
    #Obtain all the predictions after last epoch
    if learn_features and model_key == "GAE":
        nothing = True
    else:
        test_acc, test_loss, test_top_k, test_conf_matr, test_top_1_norm = test(test_loader, labels, top_k)    

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
