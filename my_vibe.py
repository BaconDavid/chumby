import os
import sys
import math
import random
from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from ete3 import Tree, TextFace, TreeStyle, NodeStyle

def seed_torch(seed=2021):
    """
    Resets the seed for the Python environment, numpy, and torch.
    
    Parameters
    ----------
    seed : int, optional
        default is 2021
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def cophenetic_distmat(tree, names):
    """
    Calculates the all-versus-all distance matrix of a tree based on the
    cophenetic distances.
    
    Parameters
    ----------
    tree : str
        a newick-formatted tree
    names : list of str
        a list of names contained within the tree. the order of the names provided
        in the list will be used to determine the order of the output.
    
    Returns
    -------
    cophmat : np.ndarray
        a square, symmetrical distance matrix
    """
    tree = treeswift.read_tree_newick(tree) if type(tree) is str else tree
    cophdic = tree.distance_matrix()
    node2name = {i:i.get_label() for i in cophdic.keys()}
    unique_names = set(node2name.values())
    assert len(node2name)==len(set(node2name))
    cophdic = {node2name[k1]:{node2name[k2]:cophdic[k1][k2] for k2 in cophdic[k1]} for k1 in cophdic.keys()}
    assert all(i in unique_names for i in names)
    cophmat = np.zeros([len(names)]*2)
    for ni, i in enumerate(names):
        for nj, j in enumerate(names[:ni]):
            cophmat[ni][nj] = cophmat[nj][ni] = cophdic[i][j]
    return cophmat

def get_bipartitions(tree, names):
    """
    Identify all bipartitions within a tree.
    
    Parameters
    ----------
    tree : ete3.Tree
        a tree object
    names : list of str
        a list of names contained within the tree. the order of the names provided
        in the list will be used to determine the order of the output.
    
    Returns
    -------
    bits : np.ndarray
        bipartition array of size (c, n) where c is each clade with more than one
        member on the tree and (n) is all taxa names on the tree. a bipartition
        for a given clade is represented based on which name is in the clade or not,
        0 if present and 1 if not present
    nodes : np.ndarray of ete3.Tree
        nodes corresponding to the (c) bipartitions
    """
    tips = set(i.name for i in tree.get_leaves())
    assert all(i in tips for i in names)
    
    n2i = {i: n for n, i in enumerate(names)}
    generator  = (n for n in tree.traverse("postorder") if not n.is_leaf())
    
    size  = len(names)
    bits  = np.zeros((size, size), dtype=np.int)
    nodes = np.zeros((size), dtype=object)
    
    for n, node in enumerate(generator):
        mask = [n2i[i.name] for i in node.get_leaves() if i.name in n2i]
        bits[n, mask] = 1
        if bits[n, 0] == 1:
            bits[n] = 1 - bits[n]
        nodes[n] = node
    
    bitsum  = bits.sum(1)
    include = (bitsum > 1) * (bitsum < size-1)
    
    return bits[include], nodes[include]

def get_support(reference_bits, sample_tree, names):
    """
    Get the branch support of a reference tree, given its bipartitions.
    
    Parameters
    ----------
    reference_bits : np.ndarray
        bipartition array of size (c, n) where c is each clade with more than one
        member on the tree and (n) is all taxa names on the tree. a bipartition
        for a given clade is represented based on which name is in the clade or not,
        0 if present and 1 if not present
    sample_tree : ete3.Tree
        a replicate tree
    names : list of str
        a list of names contained within the tree. the order of the names provided
        in the list will be used to map to the bipartitions.
    
    Returns
    -------
    out : np.ndarray
        for each reference bipartition, 1 if the bipartition is in the sample and
        0 if the bipartition is not
    """
    sample_bits = get_bipartitions(sample_tree, names)[0]
    return cdist(reference_bits, sample_bits, metric='cityblock').min(1) == 0

class CyclicalAnnealingSchedules:
    """
    Cyclic annealing schedules to help mitigate KL vanishing. This helps
    with training VAEs. We specifically use cosine annealing.
    
    Adapted from:
    https://github.com/haofuml/cyclical_annealing
    """
    @staticmethod
    def frange(start, stop, step, n_epoch):
        L = np.empty(n_epoch)
        L.fill(stop)
        v , i = start , 0
        while v <= stop:
            L[i] = v
            v += step
            i += 1
        return L
    
    @staticmethod
    def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.empty(n_epoch)
        L.fill(stop)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule
        for c in range(n_cycle):
            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L    
    
    @staticmethod
    def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.empty(n_epoch)
        L.fill(stop)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # step is in [0,1]
        # transform into [-6, 6] for plots: v*12.-6.
        for c in range(n_cycle):
            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
                v += step
                i += 1
        return L    

    @staticmethod
    def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.empty(n_epoch)
        L.fill(stop)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # step is in [0,1]
        # transform into [0, pi] for plots: 
        for c in range(n_cycle):
            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
                v += step
                i += 1
        return L

class TriangleSectorLoss:
    """
    An optimized torch implementation of TS-SS distance that is compatible with
    gradient calculations, allow it to be used as a loss function.
    
    For more information on TS-SS:
    https://github.com/taki0112/Vector_Similarity
    """ 
    def __init__(self, pseudotheta=10.):
        self._cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        self._euclidean = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=False)
        self._pseudotheta = torch.deg2rad(torch.tensor(pseudotheta))
        self._factor = torch.tensor(math.pi / 360. / 2.)
        
    def __call__(self, x1, x2):
        theta = self._pseudotheta + torch.acos(self._cosine(x1, x2))
        x1_norm = torch.linalg.norm(x1, ord=2, axis=1) # vector size
        x2_norm = torch.linalg.norm(x2, ord=2, axis=1) # vector size
        _triangle = torch.abs(torch.sin(theta)) * x1_norm * x2_norm 
        magnitude_diff = abs(x1_norm - x2_norm)
        _sector = (self._euclidean(x1, x2) + magnitude_diff).pow(2)  * theta
        return _triangle * _sector * self._factor

class LossFunction(nn.Module):
    """
    A customized torch loss function with three components:
        TSE : TS-SS error. For more info, see TriangleSectorLoss.
        MSE : Mean squared error.
        KLD : Kullbeck-Leibler divergence.
        
    Inherits from nn.Module
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """
    def __init__(self):
        super(LossFunction, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.tse = TriangleSectorLoss()

    def forward(self, x_recon, x, mu, logvar):
        mse_loss = self.mse(x_recon, x)
        tse_loss = self.tse(x_recon, x).sum()
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return tse_loss, kld_loss, mse_loss
    
class VariationalAutoencoder(nn.Module):
    """
    A simple VAE model.
    
    Inherits from nn.Module
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    
    Parameters
    ----------
    encoder_layers : list of int
        The layer sizes for the VAE's encoder. The size of the first layer
        should be the same as the input size, while the size of the last
        layer should be the same as the size of the latent dimension.
    latent_dim : int
        The size of the latent space. 
    decoder_layers : list of int
        The layer sizes for the VAE's decoder. Should be the reverse of the
        encoder layers.
    """    
    def __init__(self, encoder_layers, latent_dim, decoder_layers):
        super(VariationalAutoencoder, self).__init__()
    
        # encoder layers
        layers = zip(encoder_layers, encoder_layers[1:])
        layers = (self._add_layer(D_in, D_out) for D_in, D_out in layers)
        self.encoder = nn.Sequential(*layers)
        
        # latent vectors mu & sigma
        self.latent = self._add_layer(encoder_layers[-1], latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)
        
        # sampling vector epsilon
        self.fc3 = self._add_layer(latent_dim, latent_dim)
        self.fc4 = self._add_layer(latent_dim, decoder_layers[0])
        self.relu = nn.ReLU()
        
        # decoder layers
        layers = zip(decoder_layers, decoder_layers[1:])
        layers = (self._add_layer(D_in, D_out) for D_in, D_out in layers)
        self.decoder = nn.Sequential(*layers)
        del self.decoder[-1][-1]
    
    def _add_layer(self, D_in, D_out):
        layers = (nn.Linear(D_in, D_out),
            nn.BatchNorm1d(num_features=D_out),
            nn.ReLU())
        return nn.Sequential(*layers)
        
    def encode(self, x):
        fc1 = F.relu(self.latent(self.encoder(x)))
        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        return r1, r2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        fc3 = self.relu(self.fc3(z))
        fc4 = self.relu(self.fc4(fc3))
        return self.decoder(fc4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VIBE:
    """
    VIBE stands for VAE implemented branch support estimation.
    
    In order to perform a VIBE check on an embedding tree, resample the
    latent space of the underlying data.
    
    encoder_layers : list of int
        The layer sizes for the VAE's encoder. The size of the first layer
        should be the same as the input size, while the size of the last
        layer should be the same as the size of the latent dimension.
    latent_dim : int
        The size of the latent space. 
    decoder_layers : list of int
        The layer sizes for the VAE's decoder. Should be the reverse of the
        encoder layers.
    tse_weight : float     = 0.1,
        weight of the TSE loss term
    max_iter : int         = 10000,
        maximum number of training iterations
    patience : int         = 500,
        if early stopping is enable, this is the maximum number of training
        iterations without improvement before stopping.
    warm_up : float        = 0.1,
        percentage of the total cycles to use for warm-up
    cool_down : float      = 0.1,
        percentage of the total cycles to use for cool-down
    early_stop : bool      = True,
        toggle early stopping if exceeded patience
    kld_annealing : bool   = True,
        toggle a cosine annealing schedule for the weight of the KLD loss term
    start_beta : float     = 0.001,
        weight of the KLD loss term during warm-up / cool-down
    stop_beta : float      = 0.1,
        weight of the KLD loss term during training
    n_cycle : int          = 10,
        number of cosine annealing cycles to apply
    threads        = 22,
        if using cpu, try running with multiple threads
    device : str
        torch device for generating embeddings.
        examples: "cpu", "cuda", "cuda:0", "cuda:1"
    seed : int
        random seed
    log_dir : str
        directory to store checkpoints
    """
    def __init__(self,
                 encoder_layers = [1280, 640, 640, 320],
                 latent_dim     = 320,
                 decoder_layers = [320, 640, 640, 1280],
                 tse_weight     = 0.1,
                 max_iter       = 10000,
                 patience       = 500,
                 warm_up        = 0.1,
                 cool_down      = 0.1,
                 early_stop     = True,
                 kld_annealing  = True,
                 start_beta     = 0.001,
                 stop_beta      = 0.1,
                 n_cycle        = 10,
                 threads        = 22,
                 device         = 'cuda',
                 seed           = 2021,
                 log_dir        = '',
                ):
        
        # model architecture
        self.encoder_layers = encoder_layers
        self.latent_dim     = latent_dim
        self.decoder_layers = decoder_layers
        
        # training parameters
        self.tse_weight     = tse_weight
        self.max_iter       = max_iter
        self.patience       = patience
        
        if type(warm_up)==float:
            self.warm_up    = int(warm_up * max_iter)
        elif type(warm_up)==int:
            self.warm_up    = warm_up
            
        if type(cool_down)==float:
            self.cool_down  = int(cool_down * max_iter)
        elif type(cool_down)==int:
            self.cool_down  = cool_down
        
        self.early_stop     = early_stop
        self.kld_annealing  = kld_annealing
        self.start_beta     = start_beta
        self.stop_beta      = stop_beta
        self.n_cycle        = n_cycle
        
        # resources
        self.device         = device
        self.threads        = threads
        if log_dir.startswith('/'):
            self.log_dir = log_dir
        else:
            self.log_dir = os.getcwd()+log_dir
        
        # seed
        self.seed           = seed
        seed_torch(seed=self.seed)

    def _initialize(self, X, model_state=None, optimizer_state=None, history=None):
        ################################## error checking
        if X.shape[1] != self.encoder_layers[0]:
            raise Exception('First layer of the VAE encoder needs to match the number of input features.')
            
        if X.shape[1] != self.decoder_layers[-1]:
            raise Exception('Last layer of the VAE decoder needs to match the number of input features.')
        
        ################################## initialize history
        if optimizer_state is None:
            self.history = {
                'epoch'        : [],
                'avg_loss'     : [],
                'avg_mse_loss' : [],
                'avg_tse_loss' : [],
                'avg_kld_loss' : [],
                'beta'         : [],
            }
        else:
            items = ['epoch', 'avg_loss', 'avg_mse_loss', 'avg_tse_loss', 'avg_kld_loss', 'beta']
            assert all(i in history for i in items)
            self.history = history
        
        ################################## initialize dataset
        class MyDataset(Dataset):
            def __init__(self, X, device):
                self.X   = X
                self.X   = torch.from_numpy(self.X).to(device)
            
            def __getitem__(self, index):
                return self.X[index]
            
            def __len__(self):
                return self.X.shape[0]
            
        self.X          = X
        self.dataset    = MyDataset(self.X, self.device)
        self.dataloader = DataLoader(dataset=self.dataset, 
                                     batch_size=self.X.shape[0])
        
        ################################## initialize model
        self.model = VariationalAutoencoder(encoder_layers=self.encoder_layers,
                                            latent_dim=self.latent_dim,
                                            decoder_layers=self.decoder_layers
                                           ).to(self.device)
        
        if model_state is None:
            def init_weights_uniform_rule(model):
                classname = model.__class__.__name__
                if classname.find('Linear') != -1: # for each linear layer
                    n = model.in_features
                    y = 1.0/np.sqrt(n)
                    model.weight.data.uniform_(-y, y)
                    model.bias.data.fill_(0)

            self.model.apply(init_weights_uniform_rule)
        else:
            self.model.load_state_dict(model_state)
        
        ################################## initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
       
        if not optimizer_state is None:
            self.optimizer.load_state_dict(optimizer_state)
        
        ################################## initialize loss function
        self.loss_func = LossFunction()
        
        ################################## initialize beta (KLD weight)
        if self.kld_annealing:
            self._beta = CyclicalAnnealingSchedules.frange_cycle_cosine(
                                             start   = self.start_beta,
                                             stop    = self.stop_beta,
                                             n_cycle = self.n_cycle,
                                             n_epoch = self.max_iter-self.warm_up-self.cool_down,
                                             ratio   = 1.0)
        else:
            self._beta = [self.stop_beta] * (self.max_iter-self.warm_up-self.cool_down)
        
        return self
    
    def fit(self, *arg, handle=sys.stderr):
        """
        Fit the model to data.
        
        Parameters
        ----------
        X : np.ndarray, optional
            the data to train the model, an array with 2 axes. if using a newly 
            initialized model, the data will be passed to self._initialize and
            saved. otherwise, you can continue training a model using the saved
            data from an incomplete run.
        """
        torch.cuda.empty_cache()
        torch.set_num_threads(self.threads)

        try:
            epoch = self.history['epoch'][-1]
        except:
            self._initialize(*arg)
            epoch = 0
        
        while epoch < self.max_iter:
            epoch += 1
            self.model.train()
            self.optimizer.zero_grad()
            
            data = next(iter(self.dataloader))
            recon_batch, mu, logvar = self.model(data)
            tse_loss, kld_loss, mse_loss = self.loss_func(recon_batch, data, mu, logvar)
            
            if epoch <= self.warm_up:
                kld_weight = self.start_beta
            elif epoch > (self.max_iter - self.cool_down): 
                kld_weight = self.start_beta
            else: # cyclic annealing
                kld_weight = self._beta[epoch-self.warm_up-1]

            loss = (self.tse_weight * tse_loss) + (kld_weight * kld_loss) + mse_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            avg_loss     = loss.item() / len(self.dataloader.dataset)
            avg_tse_loss = tse_loss.item() / len(self.dataloader.dataset)
            avg_kld_loss = kld_loss.item() / len(self.dataloader.dataset)
            avg_mse_loss = mse_loss.item() / len(self.dataloader.dataset)
            
            self.history['epoch'       ] += [epoch       ]
            self.history['avg_loss'    ] += [avg_loss    ]
            self.history['avg_mse_loss'] += [avg_tse_loss]
            self.history['avg_tse_loss'] += [avg_kld_loss]
            self.history['avg_kld_loss'] += [avg_mse_loss]
            self.history['beta'        ] += [kld_weight  ]
            
            report = f'====> Epoch: {epoch} | loss: {avg_loss:.4f} | MSE: {avg_mse_loss:.4f} | TSE: {avg_tse_loss:.4f} | KLD: {avg_kld_loss:.4f} | beta: {kld_weight:.4f}'
            handle.write(f'{report}      \r')
            if epoch % 500 == 0:
                handle.write(f'\n')
            
        return self
    
    def dump(self, checkpoint_file):
        """
        Dumps the current model and its parameters into a checkpoint file.
        
        Parameters
        ----------
        checkpoint_file : str
            save to this file
        """
        torch.save({
            'X'                    : self.X,
            
            'encoder_layers'       : self.encoder_layers,
            'latent_dim'           : self.latent_dim,
            'decoder_layers'       : self.decoder_layers,
            
            'tse_weight'           : self.tse_weight,
            'max_iter'             : self.max_iter,
            'patience'             : self.patience,
            'warm_up'              : self.warm_up,
            'cool_down'            : self.cool_down,
            'early_stop'           : self.early_stop,
            'kld_annealing'        : self.kld_annealing,
            'start_beta'           : self.start_beta,
            'stop_beta'            : self.stop_beta,
            'n_cycle'              : self.n_cycle,
            'seed'                 : self.seed,
            
            'model_state_dict'     : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'history'              : self.history,
            }, checkpoint_file)
    
    @staticmethod
    def load(checkpoint_file, device='cuda'):
        """
        Loads a model from a saved checkpoint file.
        
        Parameters
        ----------
        checkpoint_file : str
            load this file
        """
        checkpoint = torch.load(checkpoint_file, map_location=device)
        
        vibe = VIBE(
            encoder_layers  = checkpoint['encoder_layers'],
            latent_dim      = checkpoint['latent_dim'],
            decoder_layers  = checkpoint['decoder_layers'],
            tse_weight      = checkpoint['tse_weight'],
            max_iter        = checkpoint['max_iter'],
            patience        = checkpoint['patience'],
            warm_up         = checkpoint['warm_up'],
            cool_down       = checkpoint['cool_down'],
            early_stop      = checkpoint['early_stop'],
            kld_annealing   = checkpoint['kld_annealing'],
            start_beta      = checkpoint['start_beta'],
            stop_beta       = checkpoint['stop_beta'],
            n_cycle         = checkpoint['n_cycle'],
            seed            = checkpoint['seed'])
        
        vibe._initialize(checkpoint['X'],
            model_state     = checkpoint['model_state_dict'],
            optimizer_state = checkpoint['optimizer_state_dict'], 
            history         = checkpoint['history'])
        
        return vibe
        
    def resample(self, n_samples):
        """
        Resamples the fixed size embeddings (self.X) using the VAE.
        
        Parameters
        ----------
        n_samples : int
            number of samples to generate.
        
        Returns
        -------
        output : np.ndarray
            for a VAE trained on a dataset of fixed size embeddings of size (n, e)
            where (n) is each example and (e) is the fixed size embedding of each
            example. returns a array of resampled embeddings of size (s, n, e) where
            (s) is the n_samples specified in the input parameters.
        """
        torch.cuda.empty_cache()
        torch.set_num_threads(self.threads)
        self.model.eval()
        X_r = np.zeros((n_samples,*self.X.shape))
        with torch.no_grad():
            for index in range(X_r.shape[0]):
                data = next(iter(self.dataloader))
                data = data.to(self.device)
                output, mu, logvar = self.model(data)
                X_r[index] = output.cpu()
        return X_r

