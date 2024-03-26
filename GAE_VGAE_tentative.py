#Now is missing the import of the data part

#Try to apply an autoencoder on the first Graph
data = train_test_split_edges(Dataset[0])

#Define Autoencoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

#Define Variational Autoencoder
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# parameters
out_channels = 2
num_features = 3
epochs = 100
variational = True

if not variational:
    # model
    model_GAE = GAE(GCNEncoder(num_features, out_channels))

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_GAE = model_GAE.to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)

    # inizialize the optimizer
    optimizer = torch.optim.Adam(model_GAE.parameters(), lr=0.01)

    def train():
        model_GAE.train()
        optimizer.zero_grad()
        z = model_GAE.encode(x, train_pos_edge_index)
        loss = model_GAE.recon_loss(z, train_pos_edge_index)
        loss.backward()
        optimizer.step()
        return float(loss)
    

    def test(pos_edge_index, neg_edge_index):
        model_GAE.eval()
        with torch.no_grad():
            z = model_GAE.encode(x, train_pos_edge_index)
        return model_GAE.test(z, pos_edge_index, neg_edge_index)

    print("Autoencoder")
    for epoch in range(1, epochs + 1):
        loss = train()

        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

else:
    model_VGAE = VGAE(VariationalGCNEncoder(num_features, out_channels))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_VGAE = model_VGAE.to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    optimizer = torch.optim.Adam(model_VGAE.parameters(), lr=0.01)

    def train():
        model_VGAE.train()
        optimizer.zero_grad()
        z = model_VGAE.encode(x, train_pos_edge_index)
        loss = model_VGAE.recon_loss(z, train_pos_edge_index)
        
        loss = loss + (1 / data.num_nodes) * model_VGAE.kl_loss()  # new line
        loss.backward()
        optimizer.step()
        return float(loss)

    def test(pos_edge_index, neg_edge_index):
        model_VGAE.eval()
        with torch.no_grad():
            z = model_VGAE.encode(x, train_pos_edge_index)
        return model_VGAE.test(z, pos_edge_index, neg_edge_index)
    
    print("Variational Autoencoder")
    for epoch in range(1, epochs + 1):
        loss = train()
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))