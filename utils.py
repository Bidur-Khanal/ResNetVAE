import os
import os.path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader as Dataloader
import torch.utils.data as data
import torchvision.datasets as datasets
import pandas as pd 
import torchvision
from torchvision import transforms
from enum import Enum
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pdb
import time 
import h5py
import random
import tifffile 
import torchvision.datasets as datasets


"""
@Author: Bidur Khanal
contains: dataset loader, ProgressMeter, and some important util functions 
util functions adopted from https://github.com/pytorch/examples/blob/main/imagenet/main.py

"""


classes = ["XR_SHOULDER","XR_HUMERUS","XR_FINGER","XR_WRIST","XR_FOREARM","XR_HAND", "XR_ELBOW"]

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

def safe_load_dict(model, new_model_state):
    """
    Safe loading of previous ckpt file.
    """
    old_model_state = model.state_dict()
    c = 0
    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'model':
            name = '.'.join(end)
        if name not in old_model_state:
            print('%s not found in old model.' % name)
            continue
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            continue
        else:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')

def load_model (net, device, ckpt_file = None, model_name = None):
    if ckpt_file is not None:
            resumed = torch.load(ckpt_file,map_location=device)

            if model_name == "moco":
                state_dict = resumed["state_dict"]
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith("module.encoder_q") and not k.startswith(
                        "module.encoder_q.fc"
                    ):
                        # remove prefix
                        state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                print("Resuming from {}".format(ckpt_file))
                safe_load_dict(net,state_dict)

            else:
        
                if 'state_dict' in resumed:
                    state_dict_key = 'state_dict'
                    print("Resuming from {}".format(ckpt_file))
                    safe_load_dict(net, resumed[state_dict_key])
                else:
                    print("Resuming from {}".format(ckpt_file))
                    safe_load_dict(net, resumed)
    return net



def oversample(dataset):

    """
    Dataset oversampler to create balanced dataset
    Args: dataset: either MURA or MEDMNIST dataset

    Function: randomly samples the data from classes to equal the total number of samples per class
              to be equal to that of one max class (with the most samples)
    """
    if hasattr(dataset, 'labels'):
        unique_labels, class_counts = np.unique(dataset.labels, return_counts=True)
        max_class = np.argmax(class_counts)
        max_sample = np.max(class_counts)
        sum = 0
        for cls in unique_labels:    
            if cls != max_class:
                add_sample = max_sample - class_counts[cls]
                sum +=add_sample
                idx = np.where(dataset.labels == cls)[0]
                sampled_idx = np.random.choice(idx, size = add_sample) 
                dataset.imgs = np.concatenate((dataset.imgs,dataset.imgs[sampled_idx]), axis = 0)
                dataset.labels = np.concatenate((dataset.labels,dataset.labels[sampled_idx]), axis = 0)
        
    elif hasattr(dataset, 'targets'):
        unique_labels, class_counts = np.unique(dataset.targets, return_counts=True)
        max_class = np.argmax(class_counts)
        max_sample = np.max(class_counts)
        sum = 0
        for cls in unique_labels:    
            if cls != max_class:
                add_sample = max_sample - class_counts[cls]
                sum +=add_sample
                idx = np.where(dataset.targets == cls)[0]
                sampled_idx = np.random.choice(idx, size = add_sample).tolist()
                samples = np.array(dataset.samples)
                targets = np.array(dataset.targets)
                
                samples = np.concatenate((samples,samples[sampled_idx]), axis = 0)
                targets = np.concatenate((targets,targets[sampled_idx]), axis = 0)

                dataset.samples = samples.tolist()
                dataset.targets = targets.tolist()
                
    else:
        raise AssertionError("Unexpected attribute isn't labels or targest!")


def undersample(dataset):

    """
    Dataset undersampler to create balanced dataset
    Args: dataset: either MURA or MEDMNIST dataset

    Function: randomly samples the data from classes to equal the total number of samples per class
              to be equal to that of one min class (with the least samples)
    """
    if hasattr(dataset, 'labels'):
        unique_labels, class_counts = np.unique(dataset.labels, return_counts=True)
        min_class = np.argmin(class_counts)
        min_sample = np.min(class_counts)

        img_size = list(dataset.imgs.shape[1:])
        img_size.insert(0,min_sample*len(unique_labels))
        label_size = list(dataset.labels.shape[1:])
        label_size.insert(0,min_sample*len(unique_labels))
        
        new_dataset_imgs = np.empty(shape = img_size, dtype = np.uint8)
        new_dataset_labels = np.empty(shape = label_size, dtype = np.uint8)
    
        sum = 0
        for cls in unique_labels:    
        
            idx = np.where(dataset.labels == cls)[0]
            sampled_idx = np.random.choice(idx, size = min_sample) 
            new_dataset_imgs[sum:sum+min_sample] = dataset.imgs[sampled_idx]
            new_dataset_labels[sum:sum+min_sample] = dataset.labels[sampled_idx]

            sum += min_sample
        dataset.imgs = new_dataset_imgs
        dataset.labels = new_dataset_labels
       
        
    elif hasattr(dataset, 'targets'):
        raise AssertionError("TODO not implemented yet fro MURA!")           
    else:
        raise AssertionError("Unexpected attribute isn't labels or targest!")


def get_acc(true_labels, predicted_labels):
    """Args: 
        true_labels: given groundtruth labels
        predicted_labels: predicted labels from the model

    Returns:
        per-class-accuracies(list): a list of per-class accuracies
        average-per-class accuracy: average of per-class accuracies
    """
    
    cm = confusion_matrix(true_labels, predicted_labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_acc = cm.diagonal()
    avg_acc =  np.mean(class_acc)
    overall_acc = accuracy_score(true_labels, predicted_labels)
    return class_acc, avg_acc, overall_acc


def computeAUROC(dataGT, dataPRED, classCount):
    # Computes area under ROC curve 
    # dataGT: ground truth data
    # dataPRED: predicted data
    outAUROC = []
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
        except ValueError:
            outAUROC.append(0)
    
    return outAUROC



# compute the AUC per class 
# implementation adapted from 
# https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class
def class_report(y_true, y_pred, y_score=None, average='micro', sklearn_cls_report = True):

    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    if sklearn_cls_report:
       
        class_report_df = pd.DataFrame(classification_report(y_true= y_true, y_pred = y_pred, output_dict=True)).transpose()
    else:

        lb = LabelBinarizer()

        if len(y_true.shape) == 1:
            lb.fit(y_true)

        #Value counts of predictions
        labels, cnt = np.unique(
            y_pred,
            return_counts=True)
        n_classes = len(labels)
        pred_cnt = pd.Series(cnt, index=labels)

        metrics_summary = precision_recall_fscore_support(
                y_true=y_true,
                y_pred=y_pred,
                labels=labels)

        avg = list(precision_recall_fscore_support(
                y_true=y_true, 
                y_pred=y_pred,
                average='weighted'))

        metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
        class_report_df = pd.DataFrame(
            list(metrics_summary),
            index=metrics_sum_index,
            columns=labels)

        support = class_report_df.loc['support']
        total = support.sum() 
        class_report_df['avg / total'] = avg[:-1] + [total]

        class_report_df = class_report_df.T
        class_report_df['pred'] = pred_cnt
        class_report_df['pred'].iloc[-1] = total

        if not (y_score is None):

            try:
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for label_it, label in enumerate(labels):
                    fpr[label], tpr[label], _ = roc_curve(
                        (y_true == label).astype(int), 
                        y_score[:, label_it])

                    roc_auc[label] = auc(fpr[label], tpr[label])

                if average == 'micro':
                    if n_classes <= 2:
                        fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                            lb.transform(y_true).ravel(), 
                            y_score[:, 1].ravel())
                    else:
                        fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                                lb.transform(y_true).ravel(), 
                                y_score.ravel())

                    roc_auc["avg / total"] = auc(
                        fpr["avg / total"], 
                        tpr["avg / total"])

                elif average == 'macro':
                    # First aggregate all false positive rates
                    all_fpr = np.unique(np.concatenate([
                        fpr[i] for i in labels]
                    ))

                    # Then interpolate all ROC curves at this points
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in labels:
                        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

                    # Finally average it and compute AUC
                    mean_tpr /= n_classes

                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr

                    roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

                class_report_df['AUC'] = pd.Series(roc_auc)

            except:
                return class_report_df

    return class_report_df


class MURA(data.Dataset):
    """MURA Dataset Object
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        seed: random seed for shuffling classes or instances (default=10)
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, num_classes= 7, seed=1):

        
        if train:
           image_folder = os.path.join(root,'train')
           path_list = pd.read_csv(os.path.join(root,'train_image_paths.csv')).values.tolist()
        else:
           image_folder = os.path.join(root,'valid')
           path_list = pd.read_csv(os.path.join(root,'valid_image_paths.csv')).values.tolist()

        self.root = root
        self.loader = default_loader
        self.samples = [pth[0] for pth in path_list]
        self.targets = [classes.index(pth[0].split('/')[2]) for pth in path_list]
        self.transform = transform
        self.target_transform = target_transform

        
        ### select only top 100 examples of each class, this is done for debugging only
        # all_targets = np.unique(self.targets)
        # curated_path_list = []
        # curated_target_list =[]
        # samples = np.array(self.samples)
        # targets = np.array(self.targets)
        # for i in all_targets:
        #     curated_path_list.extend(samples[np.where(targets == i)][0:100])
        #     curated_target_list.extend(targets[np.where(targets == i)][0:100])
        # self.samples = curated_path_list 
        # self.targets = curated_target_list

       
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        index = int(index)
        fpath, target = self.samples[index], self.targets[index]
        sample = self.loader(fpath)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class subset(data.Dataset):

    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
       
    Adopted from: https://discuss.pytorch.org/t/attributeerror-subset-object-has-no-attribute-targets/66564/4
    """

    def __init__(self, dataset, indices, args = None):
        self.dataset = dataset
        self.indices = indices
        self.mode = self.dataset.mode
        self.transform = self.dataset.transform
        self.target_transform = self.dataset.target_transform
        self.images = self.dataset.images[self.indices]
        self.targets = self.dataset.targets[self.indices]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.dataset.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.mode == "train":
            return image, target, index
        else:
            return image, target

    def __len__(self):
        return len(self.indices)

#### function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #transpose the matrix to make x-axis True Class and Y-axis Predicted Class
    cm= np.transpose(cm)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

   
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[0]),
           yticks=np.arange(cm.shape[1]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           
           #here we are not printing the title
           #title=title,
           xlabel='True label',
           ylabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig,ax



class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class custom_cifar10(torchvision.datasets.CIFAR10):

    ''' use this to create a rotation as pretask combined with the original task '''
    def __init__(self, root, train, transform = None, target_transform = None, download = True):
        super().__init__(root, train, transform, download = True)

        ### select only top certain examples of each class, this is done for debugging only
        # all_targets = np.unique(self.targets)
        # curated_path_list = []
        # curated_target_list =[]
        # samples = np.array(self.data)
        # targets = np.array(self.targets)


        # select_sample_num = 1000

        # for i in all_targets:
        #     curated_path_list.extend(samples[np.where(targets == i)][0:select_sample_num])
        #     curated_target_list.extend(targets[np.where(targets == i)][0:select_sample_num])

        # self.data = curated_path_list 
        # self.targets = curated_target_list
       
    def __getitem__(self, index):


        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img,target,index

    def __len__(self):
        return len(self.data)


'''previous method to make the cifar10 imbalanced'''
# class imbalanced_custom_cifar10(torchvision.datasets.CIFAR10):

#     ''' use this to create a rotation as pretask combined with the original task '''
#     def __init__(self, root, train, transform = None, target_transform = None, download = True):
#         super().__init__(root, train, transform, download = True)

#         ### select only top 100 examples of each class, this is done for debugging only
#         all_targets = np.unique(self.targets)
#         curated_path_list = []
#         curated_target_list =[]
#         samples = np.array(self.data)
#         targets = np.array(self.targets)

#         least_examples = 50

#         for i in all_targets:
#             curated_path_list.extend(samples[np.where(targets == i)][0:(i+1)*least_examples])
#             curated_target_list.extend(targets[np.where(targets == i)][0:(i+1)*least_examples])

#         self.data = curated_path_list 
#         self.targets = curated_target_list
       
        
#     def __getitem__(self, index):


#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.targets[index]

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)



#         return img,target,index

#     def __len__(self):
#         return len(self.data)



## adopted from 
# https://github.com/YyzHarry/imbalanced-semi-self/blob/master/dataset/imbalance_cifar.py
class imbalanced_custom_cifar10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(imbalanced_custom_cifar10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor = 0.1)
        self.gen_imbalanced_data(img_num_list)
        self.pick_randomly()

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)

       
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def pick_randomly(self):

        # lets set how many sample to pick
        samples = 10000
        
        # use this constant seed
        np.random.rand(0)
        indx = np.arange(len(self.targets))
        np.random.shuffle(indx)
        self.targets = np.array(self.targets)[indx][0:samples]
        self.data = np.array(self.data)[indx][0:samples]

    def __getitem__(self, index):


        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)



        return img,target,index

    def __len__(self):
        return len(self.data)


chexpert_classes = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema",
            "Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture",
            "Support Devices"]


class CheXpert_faster(data.Dataset):
    """CheXpert Dataset Object
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        seed: random seed for shuffling classes or instances (default=10)
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root = "D:/chest_xray/", train=True, transform=None, target_transform=None, num_classes= 14, seed=1):

        self.root = root
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"CheXpert-v1.0-small/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.targets[self.targets == -1] = 0  #label all the negative labels as 0
        self.transform = transform
        self.target_transform = target_transform
        self.pick_randomly()
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        index = int(index)
        image, target = self.images[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, index

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def pick_randomly(self):

        # lets set how many sample to pick
        samples = 10000
        # use this constant seed
        np.random.rand(0)
        indx = np.arange(len(self.targets))
        np.random.shuffle(indx)
        self.targets = np.array(self.targets)[indx][0:samples]
        self.images = np.array(self.images)[indx][0:samples]


class custom_cifar10_with_rotation_pretext(torchvision.datasets.CIFAR10):

    ''' use this to create a rotation as pretask combined with the original task '''
    def __init__(self, root, train, transform = None, target_transform = None, download = True, rotate_pretext = True, rot_angles = 4):
        super().__init__(root, train, transform, download = True)
        self.rotate_pretext = rotate_pretext
        self.rot_angles = rot_angles
        self.train = train


        # if self.train:
        #     num_samples = 1000

        #     ### select only top 100 examples of each class, this is done for debugging only
        #     all_targets = np.unique(self.targets)
        #     curated_path_list = []
        #     curated_target_list =[]
        #     samples = np.array(self.data)
        #     targets = np.array(self.targets).astype(np.int64)
           

        #     for i in all_targets:
        #         curated_path_list.extend(samples[np.where(targets == i)][0:num_samples])
        #         curated_target_list.extend(targets[np.where(targets == i)][0:num_samples])

        #     self.data = curated_path_list 
        #     self.targets = curated_target_list
        
    def __getitem__(self, index):


        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.rotate_pretext is not None:
            if self.rot_angles == 4:
                rotations = [0,1,2,3]
                random.shuffle(rotations)

                if self.train:
                    return img, torch.rot90(img, rotations[0], [1,2]), target, rotations[0], index
                else: 
                    return img, torch.rot90(img, rotations[0], [1,2]), target, rotations[0]
                    
            elif self.rot_angles == 2:
                rotations = [0,1]
                random.shuffle(rotations)
                if self.train:
                    return img, torch.rot90(img, rotations[0], [1,2]), target, rotations[0], index
                else: 
                    return img, torch.rot90(img, rotations[0], [1,2]), target, rotations[0]
        else:

            if self.train:
                return img,target, index
            else:
                return img,target

    def __len__(self):
        return len(self.data)
    

def pil_loader_tifffile(path):
    with tifffile.TiffFile(path) as tif:
        for page in tif.pages:
            img = page.asarray()
        img = Image.fromarray(img)
    return img


class custom_histopathology_classic(datasets.ImageFolder):

    """histopathology Dataset object: https://zenodo.org/record/1214456#.ZBf4GnbMKck
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """
    def __init__(self, root, transform, train = True):
        super().__init__(root, transform)
        self.root = root 
        self.transform = transform
        self.train = train
        self.loader = pil_loader_tifffile

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        #print (path,target)
        sample = self.loader(path)
       
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return sample, target, index
        else:
            return sample, target
        
    def __len__(self) -> int:
        return len(self.samples)
    



class custom_COVID19_Xray_faster(data.Dataset):
    """COVID-QU-Ex Dataset object
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        seed: random seed for shuffling classes or instances (default=10)
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root = "data/", train=True, transform=None, target_transform=None, num_classes= 3, seed=1):

        self.root = root
        self.as_rgb = True
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"COVID-QU-Dataset/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    

class custom_COVID19_Xray_with_rotation_pretext(custom_COVID19_Xray_faster):


    ''' use this to create a rotation as pretask combined with the original task '''
    def __init__(self, root ='data/', train = True, transform = None, rotate_pretext = True, rot_angles = 4):
        super().__init__(root, train, transform)
        self.rotate_pretext = rotate_pretext
        self.rot_angles = rot_angles
        self.train = train
       
    def __getitem__(self, index):


        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.rotate_pretext is not None:
            if self.rot_angles == 4:
                rotations = [0,1,2,3]
                random.shuffle(rotations)
                return image, torch.rot90(image, rotations[0], [1,2]), target, rotations[0]
            
            elif self.rot_angles == 2:
                rotations = [0,1]
                random.shuffle(rotations)
                return image, torch.rot90(image, rotations[0], [1,2]), target, rotations[0]
        else:
            if self.train:
                return image,target, index
            else:
                return image, target

    def __len__(self):
        return len(self.targets)
    

    

class custom_histopathology_faster(data.Dataset):
    """histopathology Dataset object: https://zenodo.org/record/1214456#.ZBf4GnbMKck
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self, root = "data/", train=True, transform=None, target_transform=None, num_classes= 9, seed=1):

        self.root = root
        self.as_rgb = True
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"histopathology/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        ### select only top 100 examples of each class, this is done for debugging only
        # all_targets = np.unique(self.targets)
        # curated_path_list = []
        # curated_target_list =[]
        # images = np.array(self.images)
        # targets = np.array(self.targets)
        # for i in all_targets:
        #     curated_path_list.extend(images[np.where(targets == i)][0:1000])
        #     curated_target_list.extend(targets[np.where(targets == i)][0:1000])
        # self.images = curated_path_list 
        # self.targets = curated_target_list

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    

class custom_FETAL_PLANE_faster(data.Dataset):
    """
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self, root = "data/", train=True, transform=None, target_transform=None, num_classes= 6, seed=1):

        self.root = root
        self.as_rgb = True
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"Fetal_plane/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class custom_dermnet_faster(data.Dataset):
    """
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self, root = "data/", train=True, transform=None, target_transform=None, num_classes= 3, seed=1):

        self.root = root
        self.as_rgb = True
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"Dermnet/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    


class MURA_faster(data.Dataset):
    """COVID-QU-Ex Dataset object
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        seed: random seed for shuffling classes or instances (default=10)
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root = "data/", train=True, transform=None, target_transform=None, num_classes= 3, seed=1):

        self.root = root
        self.as_rgb = True
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"MURA/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]


        # ### select only top 100 examples of each class, this is done for debugging only
        # all_targets = np.unique(self.targets)
        # curated_path_list = []
        # curated_target_list =[]
        # images = np.array(self.images)
        # targets = np.array(self.targets)
        # for i in all_targets:
        #     curated_path_list.extend(images[np.where(targets == i)][0:1000])
        #     curated_target_list.extend(targets[np.where(targets == i)][0:1000])
        # self.images = curated_path_list 
        # self.targets = curated_target_list

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class custom_histopathology_with_rotation_pretext(custom_histopathology_faster):


    ''' use this to create a rotation as pretask combined with the original task '''
    def __init__(self, root ='data/', train = True, transform = None, rotate_pretext = True, rot_angles = 4):
        super().__init__(root, train, transform)
        self.rotate_pretext = rotate_pretext
        self.rot_angles = rot_angles
        self.train = train
       
    def __getitem__(self, index):


        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.rotate_pretext is not None:
            if self.rot_angles == 4:
                rotations = [0,1,2,3]
                random.shuffle(rotations)
                return image, torch.rot90(image, rotations[0], [1,2]), target, rotations[0]
            
            elif self.rot_angles == 2:
                rotations = [0,1]
                random.shuffle(rotations)
                return image, torch.rot90(image, rotations[0], [1,2]), target, rotations[0]
                
        else:
            if self.train:
                return image,target, index
            else:
                return image, target

    def __len__(self):
        return len(self.targets)


