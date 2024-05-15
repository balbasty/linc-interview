import torch
import warnings
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
warnings.filterwarnings("ignore") 


class TrainerModule(pl.LightningModule):
    """
        Lightning model used for training

        Args:
            model: Model used for training
            trainset: training dataloader
            evalset: validation dataloader
            testset: testing dataloader
            device: device used for training ("cuda" or "cpu")
    """

    def __init__(self, model, trainset, evalset, testset, lr, lam, device="cuda"):
                 
        super(TrainerModule, self).__init__()

        self.trainset = trainset
        self.evalset = evalset
        self.testset = testset
        self.model = model
        self.lr = lr
        self.lam = lam
        
    def forward(self,x):
        return self.model(x)
    
    def calculate_loss(self,batch):
        """
        Function used on both training and validation

        Args:
        batch: training/validation batch

        Return:
        loss value
        """
        x = batch   
        disp = self(x.float())
        #defining x[:,0] as fixed img and x[:,1] as mov img
        fix = x[:,0]
        mov = x[:,1]
        loss = self.model.loss(fix, mov, disp, self.lam)
        return loss

    def training_step(self, batch, nbatch):  
        """
        Training step

        Args:
        batch: training batch

        Return:
        training loss value
        """   
        loss = self.calculate_loss(batch)              
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)  
        return loss
    
    def validation_step(self, batch, nbatch):   
        """
        Validation step

        Args:
        batch: validation batch

        Return:
        validation loss value
        """   
        loss = self.calculate_loss(batch)
        self.log("val_loss", loss, prog_bar=True) 
        return loss

    def predict_step(self, batch, nbatch):   
        """
        test step

        Args:
        batch: test batch

        Return:
        loss: test metric 
        deformed: final output (moved image)
        """   
        x = batch 
        disp = self(x)
        #calculate the final deformed img using disp and mov img
        deformed = self.model.deform(x[:,1,:,:], disp)
        loss = self.calculate_loss(batch)  
        return deformed, loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr
            )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.trainset
    def val_dataloader(self):
        return self.evalset
    def test_dataloader(self):
        return self.testset