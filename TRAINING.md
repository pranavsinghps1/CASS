# Training

Please check [INSTALL.md](INSTALL.md) for installation instructions first.

The file CASS.ipynb contains the code for self-supervised training and then loafin those weights for labelled finetunning.

```
train_csv_path = '/scratch/ps4364/BTMRI/data/train.csv'
train_imgs_dir = '/scratch/ps4364/BTMRI/data/Training/'

```
Update the train_csv_path to provide the path of the reflect the path of CSV file containg the image-label pairs.
train_imgs_dir is where the images have been stored.

```
label_num2str = {0: 'glioma',
                     1: 'pituitary',
                     2:'notumor',
                     3:'meningioma'
                     }
label_str2num = {'glioma': 0,
                     'pituitary':1,
                     'notumor':2,
                     'meningioma':3
                     }
```
label_num2str and label_str2num  shouyld be updated to match the differnt classes to integer mapping.

```
cls_weight =  [0.2, 0.5970802919708029, 1.0, 0.25255474452554744]
```
These class weights are essential for focal loss, we the normalize function defined within the notebook to get these.

```
onep_train_indices = np.random.choice(train_indices, int(len(train_indices)*0.1), replace=False)
```
Change the finetunning label fraction by specifying percentage in decimals in int(len(train_indices)*0.1).