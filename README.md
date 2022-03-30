# NYCU-Artificial-Intelligence-on-Medical-Imaging-Lab-1-
In the experiment, we are going to execute a binary classification task. Our goal is to distinguish normal people and pneumonia people from X-rays images.

Introduction:

Artificial intelligence (AI) has the capability to detect some features and output  specific labels. 
This property can be used for disease diagnosis.
It can rapidly review a large amount of data since the machine doesn’t get tired.
Moreover, some tiny lesions such as pulmonary nodules which are not easily identified.
AI can help doctors to detect it and early prevention.
This is an open dataset: Chest X-Ray on kaggle. 
The structure of this dataset is below. 
As we can see, the normal images are fewer than pneumonia images.

Workflow:

Writing our own data loader

Run different models

Fine tune some hyperparameters

See performance and discussion


#for running code

Modify your file path in lines 257 and 258.

Some hyperparameters like batch size, learning rate and epochs you can adjust in lines 249 to 251.

If you want to use data augmentation, add '+ trainset2' in line 264. By doing so, your training set will enlarge twice.

There are three models you can choose. Change model in line 278.

There are four different optimizers you can try in line 293.
