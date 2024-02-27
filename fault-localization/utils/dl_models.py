import torchvision
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.module import LightningModule

import torchmetrics as tm

def initialize_model(config):
    model_name=config["model_name"]
    channels=config["channels"] 
    num_classes=config["classes"] 
    use_pretrained=config["use_pretrained"]
    # print(f">> pretrained model {use_pretrained}")

    model_ft = None
    if "resnet" in model_name:
        if "resnet18" == model_name:
            model_ft = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        elif "resnet34" == model_name:
            model_ft = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        elif "resnet50" == model_name:
            model_ft = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        elif "resnet101" == model_name:
            model_ft = torchvision.models.resnet101(weights="IMAGENET1K_V1")
        elif "resnet152" == model_name:
            model_ft = torchvision.models.resnet152(weights="IMAGENET1K_V1")

        if channels == 1:
            model_ft.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "densenet121":
        """ Densenet
        """
        model_ft = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        if channels == 1:  
            model_ft.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
    

    elif model_name == "resnext50_32x4d":
        """ Resnext
        """
        model_ft = torchvision.models.resnext50_32x4d(weights="IMAGENET1K_V1")        
        if channels == 1:
            model_ft.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)    

    elif model_name == "vgg16":
        """ vgg16
        """
        model_ft = torchvision.models.vgg16(weights="IMAGENET1K_V1")    
        if channels == 1:
            model_ft.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft



class ImageClassifer(LightningModule):
    def __init__(self, config, model=None):
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        self.lr = config['lr']
        self.model = model      
        self.accuracy = tm.classification.MulticlassAccuracy(num_classes=10)  #tm.Accuracy(task="multiclass", num_classes=4)
        
        self.temp_accuracy = 0
        self.criterion = nn.CrossEntropyLoss()

        self.flag = True
        
        
    def forward(self, x):
        # print(x.shape)
        x = self.model(x)
        return x

    def get_loss(self, outpus, labels):
        return self.criterion(outpus, labels)    

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.get_loss(logits, y)
        preds = torch.argmax(F.log_softmax(logits, dim=1) , dim=1)

        return {'loss':loss, "preds":preds, "target":y}
    
    def training_step_end(self, outputs):
        self.accuracy(outputs["preds"],outputs["target"])
        self.temp_accuracy = self.accuracy.compute()
        self.log("train_acc", self.accuracy, prog_bar=True)
        self.log("train_loss", outputs["loss"], prog_bar=True)
        if self.flag:
            self.log("val_acc",-1.0, prog_bar=True)
            self.log("val_loss",100.0, prog_bar=True)
            self.flag = False
        


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.get_loss(logits, y)
        preds = torch.argmax(F.log_softmax(logits, dim=1) , dim=1)
        self.log('val_loss', loss)  # Important for ReduceLROnPlateau
        return {'val_loss':loss, "preds":preds, "target":y}
    
    def validation_step_end(self,outputs): # due to multi gpu training
        self.accuracy(outputs["preds"],outputs["target"])
        self.temp_accuracy = self.accuracy.compute()
        # print(self.temp_accuracy)
        self.log("val_acc", self.accuracy, prog_bar=True)
        self.log("val_loss", outputs["val_loss"], prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        x, y = batch
        logits = self(x)
        loss = self.get_loss(logits, y)
        preds = torch.argmax(F.log_softmax(logits, dim=1) , dim=1)
        return {'loss':loss, "preds":preds, "target":y}

    def test_step_end(self,outputs): # due to multi gpu training
        self.accuracy(outputs["preds"],outputs["target"])
        self.temp_accuracy = self.accuracy.compute()
        self.log("test_acc", self.accuracy, prog_bar=True)        
        self.log("test_loss", outputs["loss"], prog_bar=True)
    
    def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.config['weight_decay'])
    #     d = {
    #    'optimizer': optimizer,
    #    'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.25, patience=3, verbose=True),
    #    'monitor': 'val_loss'            
    #     }
    #     return d
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.config['weight_decay'])
        
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5),
        #     'monitor': 'val_loss',  # Name of the metric to monitor
        #     'interval': 'epoch',
        #     'frequency': 1,
        #     'strict': True,
        # }
        # return [optimizer], [scheduler]
        return optimizer
