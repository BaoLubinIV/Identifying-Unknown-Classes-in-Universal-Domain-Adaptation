import torch
import torchvision
import torchvision.transforms as T
import lightning as L
from torch.utils.data import DataLoader, Dataset
import clip
import wandb

def collate_fn(batch):
    images, labels = zip(*batch)  
    images = list(images)  
    labels = torch.tensor(labels)  
    return images, labels


class SFUniDADataModuleBase(L.LightningDataModule):
    def __init__(self, batch_size, data_dir, category_shift, train_domain, test_domain, shared_class_num,
                 source_private_class_num, target_private_class_num):
        super(SFUniDADataModuleBase, self).__init__()
        self.batch_size = batch_size
        self.train_domain = train_domain
        self.test_domain = test_domain
        self.category_shift = category_shift

        self.train_set = None
        self.test_set = None

        self.data_dir = data_dir

        self.shared_class_num = shared_class_num
        self.source_private_class_num = source_private_class_num
        self.target_private_class_num = target_private_class_num
        self.total_class_num = shared_class_num + source_private_class_num + target_private_class_num

        self.shared_classes = [i for i in range(shared_class_num)]
        self.source_private_classes = [i + shared_class_num for i in range(source_private_class_num)]
        self.target_private_classes = [self.total_class_num - 1 - i for i in range(target_private_class_num)]

        self.source_classes = self.shared_classes + self.source_private_classes
        self.target_classes = self.shared_classes + self.target_private_classes
               
        
        self.known_classes_text = []
        self.all_classes_text = []
        
    def setup(self, stage):
        self.train_set = torchvision.datasets.ImageFolder(root=self.data_dir+self.train_domain)
        self.test_set = torchvision.datasets.ImageFolder(root=self.data_dir+self.test_domain)
        
        self.all_classes_text = [k for k, v in sorted(self.test_set.class_to_idx.items(), key=lambda item: item[1])]
        
        train_indices = [idx for idx, target in enumerate(self.train_set.targets) if target in self.source_classes]
        self.train_set = torch.utils.data.Subset(self.train_set, train_indices)

        test_indices = [idx for idx, target in enumerate(self.test_set.targets) if target in self.target_classes]
        self.test_set = torch.utils.data.Subset(self.test_set, test_indices)
        
        
        idx_to_class = {v: k for k, v in self.train_set.dataset.class_to_idx.items()}  
        self.known_classes_text = [idx_to_class[i] for i in self.source_classes if i in idx_to_class]
        
        #print(f"Known classes: {self.known_classes}")
        
    def train_dataloader(self):
        if self.trainer.lightning_module.domain == 'source':
            return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
        else:
            return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                               num_workers=1, collate_fn=collate_fn)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)


class DomainNetDataModule(SFUniDADataModuleBase):
    def __init__(self, batch_size, category_shift='', train_domain='painting', test_domain='real'):
        data_dir = '../../../../data/public/DomainNet/'

        if category_shift == 'PDA':
            self.shared_class_num = 200
            self.source_private_class_num = 145
            self.target_private_class_num = 0
        elif category_shift == 'ODA':
            self.shared_class_num = 200
            self.source_private_class_num = 0
            self.target_private_class_num = 145
        elif category_shift == 'OPDA':
            self.shared_class_num = 150
            self.source_private_class_num = 50
            self.target_private_class_num = 145
        else:
            self.shared_class_num = 345
            self.source_private_class_num = 0
            self.target_private_class_num = 0

        super(DomainNetDataModule, self).__init__(batch_size, data_dir, category_shift, train_domain,
                                                  test_domain, self.shared_class_num, self.source_private_class_num,
                                                  self.target_private_class_num)


class VisDADataModule(SFUniDADataModuleBase):
    def __init__(self, batch_size, category_shift='', train_domain='train', test_domain='validation'):
        data_dir = 'data/visda/'

        train_domain = 'train'
        test_domain = 'validation'

        if category_shift == 'PDA':
            self.shared_class_num = 6
            self.source_private_class_num = 6
            self.target_private_class_num = 0
        elif category_shift == 'ODA':
            self.shared_class_num = 6
            self.source_private_class_num = 0
            self.target_private_class_num = 6
        elif category_shift == 'OPDA':
            self.shared_class_num = 6
            self.source_private_class_num = 3
            self.target_private_class_num = 3
        else:
            self.shared_class_num = 12
            self.source_private_class_num = 0
            self.target_private_class_num = 0

        super(VisDADataModule, self).__init__(batch_size, data_dir, category_shift, train_domain,
                                              test_domain, self.shared_class_num, self.source_private_class_num,
                                              self.target_private_class_num)








class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None, meta_file=None):
        self.dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
        self.wnid_to_labels = self._load_meta(meta_file) if meta_file else None

    def _load_meta(self, meta_file):
        meta_data = torch.load(meta_file)
        meta_dict = meta_data[0] 
        return {wnid: names[0] for wnid, names in meta_dict.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label_idx = self.dataset[idx]  # Load image and label index
        label_wnid = self.dataset.classes[label_idx]  # Get WordNet ID (wnid)
        label_text = self.wnid_to_labels[label_wnid] if self.wnid_to_labels else label_wnid
        return image, label_text


class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.meta_file = meta_file


    def setup(self, stage=None):
        self.train_dataset = ImageNetDataset(
            root="../../../../data/public/imagenet2012/train",
            meta_file="../../../../data/public/imagenet2012/meta.bin"
        )
        self.val_dataset = ImageNetDataset(
            root='../../../../data/public/imagenet2012/val',
            meta_file="../../../../data/public/imagenet2012/meta.bin"
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
