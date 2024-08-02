import argparse
import time
import torch.nn.functional as F
import logging
import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
from models import *
# from models_ori import *
from Clustering import Clustering
from sure_inference import both_infer
from data_loader import loader_cl, get_train_loader
import os
from kmeans_pytorch import kmeans, kmeans_predict
# import math
# import models_ori
import warnings
os.environ['OMP_NUM_THREADS']="1"

parser = argparse.ArgumentParser(description='MvCLN in PyTorch')
parser.add_argument('--data', default='5', type=int,
                    help='choice of dataset, 0-Scene15, 1-Caltech101, 2-Reuters10, 3-NoisyMNIST,'
                         '4-DeepCaltech, 5-DeepAnimal, 6-MNISTUSPS, 7-WIKI, 8-WIKI-deep, 9-NUSWIDE-deep, 10-xmedia-deep, 11-xrmb')
parser.add_argument('-li', '--log-interval', default='20', type=int, help='interval for logging info')
parser.add_argument('-bs', '--batch-size', default='1024', type=int, help='number of batch size')
parser.add_argument('-e', '--epochs', default='100', type=int, help='number of epochs to run')
parser.add_argument('-lr', '--learn-rate', default='1e-3', type=float, help='learning rate of adam')
parser.add_argument('--lam', default='0.5', type=float, help='hyper-parameter between losses')
parser.add_argument('-noise', '--noisy-training', type=bool, default=True,
                    help='training with real labels or noisy labels')
parser.add_argument('-np', '--neg-prop', default='30', type=int, help='the ratio of negative to positive pairs')
parser.add_argument('-m', '--margin', default=0.1, type=float, help='initial margin')
parser.add_argument('--q', default=4., type=float, help='q parameter')
parser.add_argument('--shift', default=7., type=float, help='initial margin')
parser.add_argument('--tau', default=.2, type=float, help='initial margin')
parser.add_argument('--ratio', default=0, type=float, help='initial test ratio')
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx to use.')
parser.add_argument('-r', '--robust', default=True, type=bool, help='use our robust loss or not')
parser.add_argument('-t', '--switching-time', default=1.0, type=float, help='start fine when neg_dist>=t*margin')
parser.add_argument('-s', '--start-fine', default=False, type=bool, help='flag to start use robust loss or not')
parser.add_argument('--settings', default=2, type=int, help='0-PVP, 1-PSP, 2-Both')
parser.add_argument('-ap', '--aligned-prop', default='0.5', type=float,
                    help='originally aligned proportions in the partially view-unaligned data')
parser.add_argument('-cp', '--complete-prop', default='1.0', type=float,
                    help='originally complete proportions in the partially sample-missing data')
parser.add_argument('--method', default='pa', type=str, help='initial margin')
parser.add_argument('--test-rate', default=0, type=float, help='initial test rate')
parser.add_argument('--NC', dest='NC', action='store_true', help='noisy correspondencel')


warnings.filterwarnings('ignore')
args = parser.parse_args()
# print("==========\nArgs:{}\n==========".format(args))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
def train(train_loader, model, criterion, optimizer, epoch, args):


    if epoch % args.log_interval == 0:
        logging.info("=======> Train epoch: {}/{}".format(epoch, args.epochs))
    model.train()
    time0 = time.time()
    ncl_loss_value = 0
    ver_loss_value = 0

    for batch_idx, (x0, x1, labels, real_labels_X, real_labels_Y, mask) in enumerate(train_loader):
       
        x0, x1, labels, real_labels = x0.to(device), x1.to(device), labels.to(device), real_labels_Y.to(device)
        if not args.NC:
            x0, x1 = x0[labels > 0], x1[labels > 0]
        
        x0 = x0.view(x0.size()[0], -1)
        x1 = x1.view(x1.size()[0], -1)
        try:
            # h0, h1, _, _ = model(x0, x1)
            h0, h1 = model(x0, x1)
        except:
            print("error raise in batch", batch_idx)


        loss = criterion(h0, h1, weight=args.NC)
        ncl_loss_value += 0
        ver_loss_value += 0
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_time = time.time() - time0


    return epoch_time





def main():                                                                
    data_name = ['Scene15', 'Caltech101', 'Reuters_dim10', 'NoisyMNIST30000', '2view-caltech101-8677sample',
                 'AWA-7view-10158sample', 'MNIST-USPS', 'wiki_2_view', 'wiki_deep_2_view', 'nuswide_deep_2_view', 'xmedia_deep_2_view', 'xrmb_2_view']
    NetSeed = 2023
    # random.seed(NetSeed)
    np.random.seed(NetSeed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(NetSeed)
    torch.cuda.manual_seed(NetSeed)
   
   
    train_pair_loader, all_loader, divide_seed = loader_cl(args.batch_size, args.neg_prop, args.aligned_prop,
                                                        args.complete_prop, args.noisy_training,
                                                        data_name[args.data])
   

    if args.data == 0:
        model = SUREfcScene().to(device)

    elif args.data == 1:
        model = SUREfcCaltech().to(device)

    elif args.data == 2:
        model = SUREfcReuters().to(device)

    elif args.data == 3:
        model = SUREfcNoisyMNIST().to(device)

    elif args.data == 4:
        model = SUREfcDeepCaltech().to(device)

    elif args.data == 5:
        model = SUREfcDeepAnimal().to(device)

    elif args.data == 6:
        model = SUREfcMNISTUSPS().to(device)

    elif args.data == 7:
        model = SUREfcWiki().to(device)

    elif args.data == 8:
        model = SUREfcWikideep().to(device)

    elif args.data == 9:
        model = SUREfcnuswidedeep().to(device)

    elif args.data == 10:
        model = SUREfcxmediadeep().to(device)

    elif args.data == 11:
        model = SUREfcxrmb().to(device)


        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    if not os.path.exists('./log/'):
        os.mkdir("./log/")
        if not os.path.exists('./log/' + str(data_name[args.data]) + '/'):
            os.mkdir('./log/' + str(data_name[args.data]) + '/')
    path = os.path.join("./log/" + str(data_name[args.data]) + "/" + 'time=' + time
                        .strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(path + '.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("******** Training begin ********")
    

    acc_list, nmi_list, ari_list = [], [], []
    train_time = 0
    # train
    
    if args.method == 'pa':      
        def pa(h0, h1, weight=False):
            cos = h0.mm(h1.t())
            sim = (cos / args.tau).exp()
            mask = torch.ones_like(sim[0]).diag() < 1
            pos = sim.diag()
            neg = sim[mask]           
            p = pos.sum() / (pos.sum() + neg.sum())
            q = args.q
            return (1. - (p ** q)) / q
        criterion = pa
    tmp = 1.
    for epoch in range(0, args.epochs + 1):
        if epoch == 0:
            with torch.no_grad():
                epoch_time = train(train_pair_loader, model, criterion, optimizer, epoch, args)
        else:
            epoch_time = train(train_pair_loader, model, criterion, optimizer, epoch, args)
        train_time += epoch_time

        
        if epoch % args.log_interval == 0:
            # print(epoch)
           
            v0, v1, gt_label, loss = both_infer(model, device, all_loader, args.settings, criterion=criterion)
            data = [v0, v1]
            
            k_means_result = []
            for i in range(1): # run times
                y_pred, ret = Clustering(data, gt_label)                
                k_means_result.append(ret)
        
            k_means_result_NMI_ave = []
            k_means_result_ARI_ave = []
            k_means_result_acc_ave = []           
            for i in range(len(k_means_result)):             
                k_means_result_NMI_ave.append(k_means_result[i]['kmeans']['NMI'])
                k_means_result_ARI_ave.append(k_means_result[i]['kmeans']['ARI'])
                k_means_result_acc_ave.append(k_means_result[i]['kmeans']['accuracy'])       
            k_means_result_NMI_ave = str(np.array(k_means_result_NMI_ave).sum(axis=0) / len(k_means_result_NMI_ave))[0:6]
            k_means_result_ARI_ave = str(np.array(k_means_result_ARI_ave).sum(axis=0) / len(k_means_result_ARI_ave))[0:6]
            k_means_result_acc_ave = str(np.array(k_means_result_acc_ave).sum(axis=0) / len(k_means_result_acc_ave))[0:6]
           

 
            logging.info("Clustering: acc= {} , nmi= {} , ari= {} , loss= {}".format(k_means_result_acc_ave, k_means_result_NMI_ave, k_means_result_ARI_ave, loss))           
            tmp = loss
         
        

                
        acc_list.append(ret['kmeans']['accuracy'])
        nmi_list.append(ret['kmeans']['NMI'])
        ari_list.append(ret['kmeans']['ARI'])
    
    

    logging.info('******** End, training time = {} s ********'.format(round(train_time, 2)))


if __name__ == '__main__':
    for i in range(1):
        main()
