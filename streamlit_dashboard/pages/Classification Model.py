import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

matchData = pd.read_excel('matchData.xlsx')
matchTestData = pd.read_excel('matchTestData.xlsx')

matchData['xG'] = matchData['xG'].fillna(matchData.groupby('Team')['xG'].transform('mean'))
matchData['xGA'] = matchData['xGA'].fillna(matchData.groupby('Team')['xGA'].transform('mean'))
matchData['shotCreatingAction'] = matchData['shotCreatingAction'].fillna(matchData.groupby('Team')['shotCreatingAction'].transform('mean'))
matchData['PassLive(LeadingtoShotAttempt)'] = matchData['PassLive(LeadingtoShotAttempt)'].fillna(matchData.groupby('Team')['PassLive(LeadingtoShotAttempt)'].transform('mean'))
matchData['PassDead(LeadingtoShotAttempt)'] = matchData['PassDead(LeadingtoShotAttempt)'].fillna(matchData.groupby('Team')['PassDead(LeadingtoShotAttempt)'].transform('mean'))
matchData['dribblesLeadingToShot'] = matchData['dribblesLeadingToShot'].fillna(matchData.groupby('Team')['dribblesLeadingToShot'].transform('mean'))
matchData['goalCreatingAction'] = matchData['goalCreatingAction'].fillna(matchData.groupby('Team')['goalCreatingAction'].transform('mean'))
matchData['PassLive(LeadingtoGoal)'] = matchData['PassLive(LeadingtoGoal)'].fillna(matchData.groupby('Team')['PassLive(LeadingtoGoal)'].transform('mean'))
matchData['PassDead(LeadingtoGoal)'] = matchData['PassDead(LeadingtoGoal)'].fillna(matchData.groupby('Team')['PassDead(LeadingtoGoal)'].transform('mean'))
matchData['dribblesLeadingToGoals'] = matchData['dribblesLeadingToGoals'].fillna(matchData.groupby('Team')['dribblesLeadingToGoals'].transform('mean'))
matchData['Shots_on_target%'] = matchData['Shots_on_target%'].fillna(matchData.groupby('Team')['Shots_on_target%'].transform('mean'))
matchData['Goals/Shot'] = matchData['Goals/Shot'].fillna(matchData.groupby('Team')['Goals/Shot'].transform('mean'))
matchData['Goals/ShotsonTarget'] = matchData['Goals/ShotsonTarget'].fillna(matchData.groupby('Team')['Goals/ShotsonTarget'].transform('mean'))
matchData['Distance_from_goal_scored'] = matchData['Distance_from_goal_scored'].fillna(matchData.groupby('Team')['Distance_from_goal_scored'].transform('mean'))
matchData['Freekick'] = matchData['Freekick'].fillna(matchData.groupby('Team')['Freekick'].transform('mean'))
matchTestData['Goals/ShotsonTarget'] = matchTestData['Goals/ShotsonTarget'].fillna(matchTestData.groupby('Team')['Goals/ShotsonTarget'].transform('mean'))

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#matchData['Wins'] = matchData['Result'].map({'W': 1, 'L': 0, 'D':0})
#matchData['Loss'] = matchData['Result'].map({'W': 0, 'L': 1, 'D':0})
#matchData['Draw'] = matchData['Result'].map({'W': 0, 'L': 0, 'D':1})

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print(matchData.head())
print(matchTestData.head())
print(matchTestData.isnull().sum())

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

extract = [3,4,5,7,8,9,13,14,15,16,17,18,19,20,21,22,23,24,25]
trainData = matchData.iloc[:, extract] 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#matchTestData['Wins'] = matchTestData['Result'].map({'W': 1, 'L': 0, 'D':0})
#matchTestData['Loss'] = matchTestData['Result'].map({'W': 0, 'L': 1, 'D':0})
#matchTestData['Draw'] = matchTestData['Result'].map({'W': 0, 'L': 0, 'D':1})

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

extract = [3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
testData = matchTestData.iloc[:, extract]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
trainData = trainData.drop("Result", axis=1)
testData = testData.drop("Result", axis=1)

trainData = trainData.apply(lambda iterator: ((iterator - iterator.mean())/iterator.std()).round(2))
testData = testData.apply(lambda x: ((x - x.mean())/x.std()).round(2))

print(trainData)
print(testData)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # WIN - 2, LOSS - 1, DRAW - 0

labelencoder = LabelEncoder()
trainData['Result'] = matchData["Result"].values
testData['Result'] = matchTestData["Result"].values


trainData['Result_cat'] = labelencoder.fit_transform(trainData['Result'])
testData['Result_cat'] = labelencoder.fit_transform(testData['Result'])

trainData = trainData.drop("Result", axis=1)
testData = testData.drop("Result", axis=1)

train_Data = trainData.iloc[:, 0:-1]
trainDataResult = trainData.iloc[:,-1]
test_Data = testData.iloc[:, 0:-1]
testDataResult = testData.iloc[:, -1]

features_length = len(train_Data.columns)

print(train_Data)
print(trainDataResult)
print(test_Data)
print(testDataResult)
#--------------------------------------------------------------------------------------------

def get_class_distribution(obj):
    count_dict = {
        "W" : 0,
        "L" : 0,
        "D" : 0
    }

    for i in obj:
        if i == 0:
            count_dict['D'] +=1
        elif i == 1:
            count_dict['L'] += 1
        elif i == 2:
            count_dict['W'] += 1
        else:
            print("Check Classes")
    return count_dict

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25,10))

#Train
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(trainDataResult)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Class Distribution in Train Set')

# Test
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(testDataResult)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Class Distribution in Test Set')


#--------------------------------------------------------------------------------------
## Neural Network
### Custom Dataset

class ClassifierDataset(Dataset):

    def __init__(self,X_data,y_data):
        self.X_data = X_data
        self.y_data = y_data
    
    def __getitem__(self, index):
        return self.X_data[index],self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_Data , trainDataResult = np.array(train_Data) , np.array(trainDataResult)

test_Data , testDataResult = np.array(test_Data) , np.array(testDataResult)



trainDataset = ClassifierDataset(torch.from_numpy(train_Data).float(), torch.from_numpy(trainDataResult).long())

testDataset = ClassifierDataset(torch.from_numpy(test_Data).float(), torch.from_numpy(testDataResult).long())

#-----------------------------------------------------------------------------------------
## Model Parameters

EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_FEATURES = features_length
NUM_CLASSES = 3

#-----------------------------------------------------------------------------------
## Data Loader

train_loader = DataLoader(dataset=trainDataset,batch_size=BATCH_SIZE)

test_loader = DataLoader(dataset=testDataset,batch_size=16)

#------------------------------------------------------------------------------------


class_count = [ i for i in get_class_distribution(trainDataResult).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float)

print(class_weights)




#----------------------------------------------------------------------------------
## Define Neural Net Architecture

class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self,x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x

#--------------------------------------------------------------------------
## Intialize the model, optimizer and Loss Function. We will use nn.CrossEntropyLoss because this
## is a MultiClassification problem.We don’t have to manually apply a log_softmax layer after our 
## final layer because nn.CrossEntropyLoss does that for us. However, we need to apply log_softmax 
## for our validation and testing.

model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters() , lr=LEARNING_RATE)

print(model)

#------------------------------------------------------------------------------
## Function to calculate Accuracy Per EPOCH
## This function takes y_pred and y_test as input arguments. 
## We then apply log_softmax to y_pred and extract the class which has a higher probability.
## After that, we compare the the predicted classes and the actual classes to calculate the accuracy.

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test)
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)
    return acc

## Accuracy per epoch and loss per epoch for train sets.

accuracy_stats = {
    'train' : []
}

loss_stats = {
    'train':[]
}

#------------------------------------------------------------------------------
## Train the Model 

print("Begin Training")
for e in (range(1,EPOCHS+1)):
    #TRAINING
    train_epoch_loss = 0
    train_epoch_accuracy = 0

    model.train()
    for x_train, y_train in  train_loader:
        x_train = x_train
        y_train = y_train

        optimizer.zero_grad()

        y_train_pred = model(x_train)
        train_loss = criterion(y_train_pred,y_train)
        train_acc = multi_acc(y_train_pred, y_train)

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_accuracy += train_acc.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        accuracy_stats['train'].append(train_epoch_accuracy/len(train_loader))

    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Train Accuracy: {train_epoch_accuracy/len(train_loader):.3f}')
