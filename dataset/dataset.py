import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FolderDataset(Dataset):
    def __init__(self, data_root, transform=None):
        super(FolderDataset, self).__init__()
        self.data_root = data_root
        self.transform = transform
        self.supported = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp']
        self.classes, self.class_to_idx = self.__find_classes(data_root)
        self.images = self.__make_dataset(data_root, self.classes, self.class_to_idx)

    def __find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    def __make_dataset(self, dir, classes, class_to_idx):
        images = []
        for cls in classes:
            for root, _, names in os.walk(os.path.join(dir,cls)):
                for name in names:
                    if os.path.splitext(name)[-1].lower() in self.supported:
                        images.append((os.path.join(root, name), class_to_idx[cls]))
        return images
        
    def __len__(self):
        return len(self.images)
    
    """
        __getitem__(self, idx)
        return: (image_cls_tensor, image_tensor)
        image_tensor: [3, H, W]
    """
    def __getitem__(self, idx):
        img_pil = Image.open(self.images[idx][0]).convert('RGB')
        img_cls = torch.tensor(self.images[idx][1], dtype=torch.int64)
        img_transforms = self.transform if self.transform else transforms.Compose([
            transforms.ToTensor(),
        ])
        img_t = img_transforms(img_pil)
        return img_cls, img_t


"""
    Image Manipulation Localization Dataset
    Folder Structure:
    |--data_root
      |--authentic/real (optional)
        |--*.[jpg | png | tif | ...]
      |--copymove
        |--fake
           |--{fake_img_name}.[jpg | png | tif | ...]
        |--mask
           |--{fake_img_name}.png
      |--splice
        |--fake
          |--...(the same as copymove)
        |--mask
          |--...(the same as copymove)
      |--...(other manipulation)
"""
class IMLDataset(Dataset):
    def __init__(self, data_root, transform=None):
        super(IMLDataset, self).__init__()
        self.data_root = data_root
        self.transform = transform
        self.supported = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp']
        self.real_labels = ['authentic', 'real']
        self.classes, self.class_to_idx = self.__find_classes(data_root)
        self.class_to_binary_idx = self.__find_binary_class(self.classes)
        self.images = self.__make_dataset(data_root, self.classes, self.class_to_binary_idx)
    
    # find folder classes
    def __find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    # folder classes to real/fake classes, 0 for real and 1 for fake
    def __find_binary_class(self, classes):
        classes = self.classes
        class_to_binary_idx = dict()
        for cls in classes:
            if cls in self.real_labels:
                class_to_binary_idx[cls] = 0
            else:
                class_to_binary_idx[cls] = 1
        return class_to_binary_idx
                
    def __make_dataset(self, data_root, classes, class_to_binary_idx):
        images = []    # element: tuples, (fake_image_cls, fake_image_path, mask_path) or (real_image_cls, real_image_path, None)
        for cls in classes:
            if class_to_binary_idx[cls] == 0:    # real case
                mask_path = None
                for root, _, names in os.walk(os.path.join(data_root, cls)):
                    for name in names:
                        if os.path.splitext(name)[-1].lower() in self.supported:
                            images.append((0, os.path.join(root, name), mask_path))
            else:    # fake case
                fake_names = os.listdir(os.path.join(data_root, cls, 'fake'))
                # mask_names = os.listdir(os.path.join(data_root, cls, 'mask'))
                for fake_name in fake_names:
                    if os.path.splitext(fake_name)[-1].lower() in self.supported:
                        mask_name = f'{os.path.splitext(fake_name)[0]}.png'
                        # # check masks
                        # try:
                        #     assert(mask_name in mask_names)
                        # except:
                        #     print(f'Mask not found for {os.path.join(data_root, cls, "fake", fake_name)}.')
                        images.append((1, os.path.join(data_root, cls, 'fake', fake_name), os.path.join(data_root, cls, 'mask', mask_name),))
        return images
        
    def __len__(self):
        return len(self.images)
    
    """
        __getitem__(self, idx)
        return: (image_cls_tensor, image_tensor, mask_tensor)
        image_tensor: [3, H, W]
        mask_tensor: [1, H, W]
    """
    def __getitem__(self, idx):
        img_cls = torch.tensor(self.images[idx][0], dtype=torch.float32)
        img_pil = Image.open(self.images[idx][1]).convert('RGB')
        img_transforms = self.transform if self.transform else transforms.Compose([
            transforms.ToTensor(),
        ])
        img_t = img_transforms(img_pil)
        mask_path = self.images[idx][2]
        if mask_path is not None:
            mask_pil = Image.open(mask_path).convert('L').resize((img_t.shape[1], img_t.shape[2]))
            mask_t = transforms.ToTensor()(mask_pil).squeeze(0)
        else:    # real case
            mask_t = torch.zeros((img_t.shape[1], img_t.shape[2]), dtype=torch.float32)
        return img_cls, img_t, mask_t
