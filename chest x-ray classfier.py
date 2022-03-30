import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from torchvision import models
# from tqdm import tqdm_notebook as tqdm
import time
from tqdm import tqdm
import warnings
import copy
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from torchsummary import summary
from sklearn.metrics import accuracy_score,classification_report, f1_score,roc_auc_score

#for confusion matrix
import seaborn as sns




def images_transforms(phase):
    if phase == 'training':
        data_transformation =transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(degrees = (-10, 20)),
            transforms.RandomVerticalFlip(p=0.7),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    else:
        data_transformation=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    
    return data_transformation



class ResNet50(nn.Module):
   def __init__(self,num_class,pretrained_option=False):
        super(ResNet50,self).__init__()
        self.model=models.resnet50(pretrained=pretrained_option)
        
        if pretrained_option==True:
            for param in self.model.parameters():
                param.requires_grad=False

        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)
        
   def forward(self,X):
        out=self.model(X)
        return out



class VGG16(nn.Module):
   def __init__(self,num_class,pretrained_option=False):
        super(VGG16,self).__init__()
        self.model=models.vgg16(pretrained=pretrained_option)
        
        if pretrained_option==True:
            for param in self.model.parameters():
                param.requires_grad=False

        self.model.classifier.add_module('add', nn.Linear(1000,2))
        
   def forward(self,X):
        out=self.model(X)
        return out


class Alex_net(nn.Module):
   def __init__(self,num_class,pretrained_option=False):
        super(Alex_net,self).__init__()
        self.model=models.alexnet(pretrained=pretrained_option)
        
        if pretrained_option==True:
            for param in self.model.parameters():
                param.requires_grad=False

        self.model.classifier.add_module('add', nn.Linear(1000,2))
        
   def forward(self,X):
        out=self.model(X)
        return out



def plot_(list,title, x_lable, y_lable):
    plt.plot(list)
    plt.title(title)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.show()



def training(model, train_loader, test_loader, Loss, optimizer, epochs, device, num_class, name):
    model.to(device)
    best_model_wts = None
    
    best_evaluated_acc = 0
    train_acc = []
    test_acc = []
    test_Recall = []
    test_Precision = []
    test_F1_score = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer , gamma = 0.98)
    for epoch in range(1, epochs+1):
        with torch.set_grad_enabled(True):
            model.train()
            total_loss=0
            correct=0
            for idx,(data, label) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                        
                data = data.to(device,dtype=torch.float)
                label = label.to(device,dtype=torch.long)

                predict = model(data)      

                loss = Loss(predict, label.squeeze())

                total_loss += loss.item()
                pred = torch.max(predict,1).indices
                correct += pred.eq(label).cpu().sum().item()
                        
                loss.backward()
                optimizer.step()

            total_loss /= len(train_loader.dataset)
            correct = (correct/len(train_loader.dataset))*100.
            print ("Epoch : " , epoch)
            print ("Loss : " , total_loss)
            print ("Correct : " , correct)
               
        scheduler.step()

        #testing
        accuracy  , Recall , Precision , F1_score = evaluate(model, device, test_loader, epoch, epochs)
        train_acc.append(correct)  
        test_acc.append(accuracy)
        test_Recall.append(Recall)
        test_Precision.append(Precision)
        test_F1_score.append(F1_score)

        if accuracy > best_evaluated_acc:
            best_evaluated_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        print('train_accuracy = {}'.format(train_acc))  
        print('test_accuracy = {}'.format(test_acc))    
        print('test_F1_score = {}'.format(test_F1_score)) 
    #save model
    torch.save(best_model_wts, name+".pt")
    model.load_state_dict(best_model_wts)
    plot_(train_acc, 'train_accuracy', 'epoch', 'train_acc')
    plot_(test_acc, 'test_accuracy', 'epoch', 'test_acc')
    plot_(test_F1_score, 'F1-Score', 'epoch', 'F1-score')
    

    return train_acc , test_acc , test_Recall , test_Precision , test_F1_score


def evaluate(model, device, test_loader, epoch, epochs):
    correct=0
    TP=0
    TN=0
    FP=0
    FN=0
    with torch.set_grad_enabled(False):
        model.eval()
        for idx,(data,label) in enumerate(test_loader):
            data = data.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.long)
            predict = model(data)
            pred = torch.max(predict,1).indices
            #correct += pred.eq(label).cpu().sum().item()
            for j in range(data.size()[0]):
                #print ("{} pred label: {} ,true label:{}" .format(len(pred),pred[j],int(label[j])))
                if (int (pred[j]) == int (label[j])):
                    correct +=1
                if (int (pred[j]) == 1 and int (label[j]) ==  1):
                    TP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  0):
                    TN += 1
                if (int (pred[j]) == 1 and int (label[j]) ==  0):
                    FP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  1):
                    FN += 1
        print ("TP : " , TP)
        print ("TN : " , TN)
        print ("FP : " , FP)
        print ("FN : " , FN)
        if epoch == epochs:
            Confusion_Matrix = np.zeros((2,2), dtype = np.int16)
            Confusion_Matrix[1][1] = TN
            Confusion_Matrix[1][0] = FP
            Confusion_Matrix[0][1] = FN
            Confusion_Matrix[0][0] = TP
            ax = sns.heatmap(Confusion_Matrix, annot=True, cmap='Blues')
            ax.set_title('Confusion Matrix \n\n')
            ax.set_xlabel('\nPredicted Values')
            ax.set_ylabel('Actual Values ')

            ax.xaxis.set_ticklabels(['True','False'])
            ax.yaxis.set_ticklabels(['True','False'])

            plt.show()



        
        print ("num_correct :",correct ," / " , len(test_loader.dataset))
        if (TP + FN == 0):
            Recall = 0
        else:
            Recall = TP/(TP+FN)
        print ("Recall : " ,  Recall )
        if(TP + FP == 0):
            Precision = 0
        else:    
            Precision = TP/(TP+FP)
        print ("Preecision : " ,  Precision )
        if (Precision + Recall == 0):
            F1_score = 0
        else:
            F1_score = 2 * Precision * Recall / (Precision + Recall)
        print ("F1 - score : " , F1_score)

        correct = (correct/len(test_loader.dataset))*100.
        print ("Accuracy : " , correct ,"%")

    return correct , Recall , Precision , F1_score

if __name__=="__main__":
    IMAGE_SIZE=(128,128)
    batch_size=128
    learning_rate = 0.005
    epochs=30
    num_classes=2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (device)

    train_path='../dataset/chest_xray/train'
    test_path='../dataset/chest_xray/test'



    trainset1=datasets.ImageFolder(train_path,transform=images_transforms('train'))
    trainset2=datasets.ImageFolder(train_path,transform=images_transforms('training'))
    trainset = trainset1 
    testset=datasets.ImageFolder(test_path,transform=images_transforms('test'))
    


    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0) 
    test_loader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=0)  
 


    model1 = ResNet50(2, True)
    model2 = VGG16(2, True)

    model3 = Alex_net(2,True)
    model = model3
    #print('-----------------resnet50-------------------')
    #print(model1)
    #print('-----------------vgg16----------------------')
    #print(model2)
    #print('-----------------alex_net--------------------')
    #print(model3)

    Loss = nn.CrossEntropyLoss()


    optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer3 = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    optimizer4 = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    optimizer = optimizer4


    train_acc , test_acc , test_Recall , test_Precision , test_F1_score  = training(model, train_loader, test_loader, Loss, optimizer,epochs, device, 2, 'CNN_chest_Alex1')






