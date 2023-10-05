import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F

from datetime import datetime
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from config import BaseConfig
from dataset import FolderDataset, random_split_dataset, IMLDataset
from models import build_model
from models.LGMNet import LGMNet
from utils.earlystop import EarlyStopping
from utils.img_processing import img_post_processing, denormalize_batch_t, denormalize_t
from utils.metrics import eval_metrics, acc_score, auc_score, f1_score
from utils.utils import set_random_seed
from utils.any2img import save_single_img


def parse_args():
    parser = argparse.ArgumentParser(description='Train Argument Parser.')
    parser.add_argument('--config', type=str, default='configs/base_train.yaml', help='Configuration for Training.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    return parser.parse_args()


def prepare_train_output_dir(config, tag):
    output_dir = os.path.join(config.train.save_dir, tag)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(os.path.join(output_dir, 'models')):
        os.makedirs(os.path.join(output_dir, 'models'))
    if not os.path.isdir(os.path.join(output_dir, 'loc')):
        os.makedirs(os.path.join(output_dir, 'loc'))
    return output_dir


def img_trans(config):
    data_config = config.train.data_transforms
    trans = []
    if data_config.crop_enabled:
        trans.append(transforms.Resize(data_config.resize))
        trans.append(transforms.RandomApply(nn.ModuleList([transforms.RandomCrop(data_config.crop)]), p=0.5))
    if data_config.flip_enabled:
        trans.append(transforms.RandomHorizontalFlip(p=0.5))
    if data_config.post_processing_enabled:
        trans.append(transforms.Lambda(lambda img: img_post_processing(img, data_config)))
    if data_config.resize_enabled:
        trans.append(transforms.Resize(data_config.resize))
    trans.append(transforms.ToTensor())
    if data_config.normalize_enabled:
        trans.append(transforms.Normalize(mean=data_config.normalize.mean, std=data_config.normalize.std))
    return trans


def load_model(model, checkpoint):
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)


def save_model(model, dir, name):
    checkpoint = {
        'model': model.state_dict()
    }
    torch.save(checkpoint, f'{os.path.join(dir, name)}')


def save_loc_img_pack(dir, tag, fake, gt, preds, mean=None, std=None):
    if mean is not None or std is not None:
        fake = denormalize_t(fake, mean, std)
    save_single_img(fake, dir, f'{tag}_fake.png', aligned=False, reshape_format='c h w -> h w c')
    save_single_img(gt, dir, f'{tag}_gt.png', aligned=False, reshape_format='c h w -> h w c')
    for i in range(len(preds)):
        pred_mask = preds[i][0]
        save_single_img(pred_mask, dir, f'{tag}_pred_{i}.png', aligned=False, reshape_format='c h w -> h w c')


if __name__ == '__main__':
    args = parse_args()
    config = BaseConfig(args.config).cfg()
    print(config)

    set_random_seed(config.train.seed)
    train_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tag = 'train-{}'.format(train_time)
    output_dir = prepare_train_output_dir(config, tag)

    train_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs', 'train'))
    test_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs', 'test'))
    val_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs', 'val'))
    with open(os.path.join(output_dir, 'logs', 'config.txt') ,'w') as f:
        f.write(str(config))

    data_trans = transforms.Compose(img_trans(config))
    imdl_loc_dataset = IMLDataset(config.train.dataset.data_root, transform=data_trans)
    train_dataset, val_dataset, test_dataset = random_split_dataset(imdl_loc_dataset, config.train.dataset.split)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.train.dataset.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, drop_last=False)
    train_data_len = len(train_dataset)
    device = 'cuda' if config.model.use_gpu else 'cpu'

    model = LGMNet()
    model = model.to(device)
    if config.model.load_from_checkpoint:
        load_model(model, os.path.join(config.model.path, config.model.checkpoint_name))
    else:
        model.init_weights()

    # early_stop = EarlyStopping(config.train.hyperparameter.early_stop, delta=0.001, verbose=True)
    # early_stop_enabled = config.train.hyperparameter.early_stop_enabled
    # early_stop_metric = config.train.hyperparameter.early_stop_metric
    # if early_stop_enabled:
    #     try:
    #         assert(early_stop_metric in config.train.metrics)
    #     except:
    #         print('Early stop metric not found in training metrics. Please add to metrics for evaluation.')
    #         exit(0)

    # contrastive_loss_enabled = config.train.hyperparameter.contrastive_loss_enabled

    # BCELogits = nn.BCEWithLogitsLoss()
    BCELoss = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(config.train.epoch):
        train_loss, train_num = 0, 0
        with tqdm(total=len(train_dataset)) as train_pbar:
            train_pbar.set_description('Training Data Processing')
            for id, data in enumerate(train_dataloader):
                model.train()
                optimizer.zero_grad()
                if args.debug:
                    if id > 20:
                        break
                cls, img_t, mask_t = data
                data_batch = cls.shape[0]
                train_num += data_batch
                if config.model.use_gpu:
                    cls = cls.to(torch.device(device))
                    img_t = img_t.to(torch.device(device))
                    mask_t = mask_t.to(torch.device(device))
                pred_masks = model(img_t)
                mask_t = torch.unsqueeze(mask_t, dim=1)
                f4, f3, f2, f1, f0, map = pred_masks
                loss_loc_4 = BCELoss(f4, F.interpolate(mask_t, size=(8, 8)))
                loss_loc_3 = BCELoss(f3, F.interpolate(mask_t, size=(16, 16)))
                loss_loc_2 = BCELoss(f2, F.interpolate(mask_t, size=(32, 32)))
                loss_loc_1 = BCELoss(f1, F.interpolate(mask_t, size=(64, 64)))
                loss_loc_0 = BCELoss(f0, F.interpolate(mask_t, size=(128, 128)))
                loss_loc_final = BCELoss(map, F.interpolate(mask_t, size=(256, 256)))
                # loss_det = BCELoss(det, cls)
                loss_loc = loss_loc_4 + loss_loc_3 + loss_loc_2 + loss_loc_1 + loss_loc_0 + loss_loc_final
                losses = loss_loc
                losses.backward()
                optimizer.step()
                if (id + 1) % 20 == 0:
                    # save localization maps
                    saved_img = img_t[0]
                    saved_mask = mask_t[0]
                    save_loc_img_pack(os.path.join(output_dir, 'loc'), f'train-{str(epoch).zfill(3)}-{str(id).zfill(6)}', saved_img, saved_mask, pred_masks, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
                if (id + 1) % 200 == 0:
                    print(f'loss loc: {loss_loc}.')
                    # train_writer.add_scalar('loss_det', loss_det, global_step = epoch * train_data_len + train_num)
                    train_writer.add_scalar('loss_loc', loss_loc, global_step = epoch * train_data_len + train_num)
                train_pbar.update(data_batch)
            torch.save(model.state_dict(), os.path.join(output_dir, 'models', '{}_{}.pth'.format(config.train.save_name, epoch)))
        # model.eval()
        # val_det_preds, val_det_labels = torch.tensor([]).to(torch.device(device)), torch.tensor([]).to(torch.device(device))
        # with tqdm(total=len(val_dataset)) as val_pbar:
        #     val_pbar.set_description('Validation Data Processing')
        #     for id, data in enumerate(val_dataloader):
        #         if id > 100:
        #             break
        #         cls, img_t, mask_t = data
        #         data_batch = cls.shape[0]
        #         if config.model.use_gpu:
        #             cls = cls.to(torch.device(device))
        #             img_t = img_t.to(torch.device(device))
        #             mask_t = mask_t.to(torch.device(device))
        #         det, pred_mask = model(img_t)
        #         val_det_preds = torch.cat([val_det_preds, det], dim=0)
        #         val_det_labels = torch.cat([val_det_labels, cls], dim=0)
        #         val_pbar.update(data_batch)
        #     val_det_preds = val_det_preds.cpu().detach().numpy()
        #     val_det_labels = val_det_labels.cpu().detach().numpy()
        #     acc = acc_score(val_det_labels, val_det_preds)
        #     auc = auc_score(val_det_labels, val_det_preds)
        #     f1 = f1_score(val_det_labels, val_det_preds)
        # val_writer.add_scalar('acc', acc, global_step = epoch * train_data_len + train_num)
        # val_writer.add_scalar('auc', auc, global_step = epoch * train_data_len + train_num)
        # val_writer.add_scalar('f1', f1, global_step = epoch * train_data_len + train_num)
# if early_stop_enabled:
        #     early_stop(val_metrics[early_stop_metric], model, os.path.join(output_dir, 'models', '{}_best.pth'.format(config.train.save_name)))

            


            # test_probs, test_labels = torch.tensor([]).to(torch.device(device)), torch.tensor([]).to(
            #     torch.device(device))
            # with tqdm(total=len(test_dataset)) as test_pbar:
            #     test_pbar.set_description('Test Dataset Processing')
            #     for id, data in enumerate(test_dataloader):
            #         if args.debug:
            #             if id > 20:
            #                 break
            #         img_t, cls = data
            #         data_batch = cls.shape[0]
            #         if config.model.use_gpu:
            #             img_t = img_t.to(torch.device(device))
            #             cls = cls.to(torch.device(device))
            #         output = model.test_batch((img_t, cls))
            #         pred = output
            #         prob = F.softmax(pred, dim=1)
            #         test_probs = torch.cat([test_probs, prob], dim=0)
            #         test_labels = torch.cat([test_labels, cls], dim=0)
            #         test_pbar.update(data_batch)
            #     test_probs = test_probs.cpu().detach()
            #     test_labels = test_labels.cpu().detach()
            # test_metrics = eval_metrics(config.train.metrics, (test_probs.numpy(), test_labels.numpy()))
            # print('Testing Metrics')
            # for k, v in test_metrics.items():
            #     test_writer.add_scalar(k, v, global_step=epoch)
            #     print('{}: {}'.format(k, v))

        print('Epoch: {0} / {1}.'.format(epoch, config.train.epoch))