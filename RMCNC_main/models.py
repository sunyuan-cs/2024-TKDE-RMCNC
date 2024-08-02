import torch.nn as nn
import torch
import torch.nn.functional as F


class SUREfcNoisyMNIST(nn.Module):
    def __init__(self):
        super(SUREfcNoisyMNIST, self).__init__()
        num_fea, mid_dim = 512, 1024
        self.encoder0 = nn.Sequential(
            nn.Linear(784, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(True),
            
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(True),
            
            
            nn.Linear(mid_dim, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )


    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1


class SUREfcCaltech(nn.Module):
    def __init__(self):
        super(SUREfcCaltech, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(1984, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            #  
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1


class SUREfcScene(nn.Module):  # 20, 59
    def __init__(self):
        super(SUREfcScene, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(20, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(59, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1

class SUREfcWiki(nn.Module):  # 20, 59
    def __init__(self):
        super(SUREfcWiki, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1

class SUREfcWikideep(nn.Module):  # 4096,300
    def __init__(self):
        super(SUREfcWikideep, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )


    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1

class SUREfcnuswidedeep(nn.Module):  # 4096,300
    def __init__(self):
        super(SUREfcnuswidedeep, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1
    
class SUREfcxmediadeep(nn.Module):  # 4096,300
    def __init__(self):
        super(SUREfcxmediadeep, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )


    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1

class SUREfcxrmb(nn.Module):  # 273,112
    def __init__(self):
        super(SUREfcxrmb, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(273, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(112, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )


    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1
      
class SUREfcReuters(nn.Module):
    def __init__(self):
        super(SUREfcReuters, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )
        self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),                                    
                                      nn.Linear(1024, 10))
        self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),                                    
                                      nn.Linear(1024, 10))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1


class SUREfcMNISTUSPS(nn.Module):
    def __init__(self):
        super(SUREfcMNISTUSPS, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )



    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1


class SUREfcDeepCaltech(nn.Module):
    def __init__(self):
        super(SUREfcDeepCaltech, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )


    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder0(x1.view(x1.size()[0], -1))
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1


class SUREfcDeepAnimal(nn.Module):
    def __init__(self):
        super(SUREfcDeepAnimal, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
 
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder0(x1.view(x1.size()[0], -1))
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        return h0, h1
