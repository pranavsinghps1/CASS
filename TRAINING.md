# Training

Please check [INSTALL.md](INSTALL.md) for installation instructions first.

The file CASS.ipynb contains the code for self-supervised training and then loafin those weights for labelled finetunning.

```
data_path = '/scratch/ISIC2019/train/isic_train.csv'
train_imgs_dir = '/scratch/ISIC2019/train/'

```
Update the data_path to provide the path of the reflect the path of CSV file containg the image-label pairs.
train_imgs_dir is where the images have been stored.

```
label_num2str and label_str2num
```
label_num2str and label_str2num  shouyld be updated to match the differnt classes to integer mapping.

```
cls_weight =  [1.0, 0.4717294571343815, .... 0.20134480371798677, 0.2]
```
These class weights are essential for focal loss, we the normalize function defined within the notebook to get these.

```
onep_train_indices = np.random.choice(train_indices, int(len(train_indices)*0.1), replace=False)
```
Change the finetunning label fraction by specifying percentage in decimals in int(len(train_indices)*0.1).