import os
from torch import autograd, optim
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from model_seg import *
from tqdm import tqdm
import PIL.Image as Image

Image.MAX_IMAGE_PIXELS = None
import platform
import argparse
sysstr = platform.system()
from dataset import *
from loss_seg import *

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def calculate_dice(pred, mask):
    eps = 1e-5
    dice = (2.0 * np.sum(mask * pred) + eps) / (np.sum(mask) + np.sum(pred) + eps)
    return dice

class AverageMeter(object):
    def __init__(self, name):
        self.name = name
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

def train(args):
    if True:
        basic_model = args.name
        best_dice = 0.0
        initial_lr = 0.01
        weight_decay = 3e-5
        Train_BATCH_SIZE = 20
        num_epochs = 100

        print("loading dataset ...............")


        patients_tr = '/mnt/hdd1/zhaoxinyi/DATA_new/Data2D_256_8c_random/imagesTr/'
        tr_list = [os.path.join(patients_tr, file_name) for file_name in os.listdir(patients_tr)]

        patients_ts = '/mnt/hdd1/zhaoxinyi/DATA_new/Data2D_256_8c_random/imagesTs/'
        ts_list = [os.path.join(patients_ts, file_name) for file_name in os.listdir(patients_ts)]

        dataset_tr = DataLoader_seg(tr_list)
        train_dataloader = DataLoader_seg(dataset_tr, batch_size=Train_BATCH_SIZE, shuffle=True)
        dataset_val = DataLoader_seg(ts_list)
        test_dataloader = DataLoader_seg(dataset_val, batch_size=1, shuffle=False)

        network = get_model_seg(inc=8)

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            network = nn.DataParallel(network)
            network.cuda().to(device)

        loss_factor = {'Dice': 1}
        Dice_criterion = DiceLoss().to(device)
        optimizer = optim.SGD(network.parameters(), initial_lr, weight_decay=weight_decay,
                              momentum=0.99, nesterov=True)

        if not os.path.exists("/mnt/hdd1/zhaoxinyi/Centermask2/save_models_seg2d/" + basic_model):
            if not os.path.exists("/mnt/hdd1/zhaoxinyi/Centermask2/save_models_seg2d/"):
                os.mkdir("/mnt/hdd1/zhaoxinyi/Centermask2/save_models_seg2d/")
            os.mkdir("/mnt/hdd1/zhaoxinyi/Centermask2/save_models_seg2d/" + basic_model)

        if not os.path.exists("/mnt/hdd1/zhaoxinyi/Centermask2/test_metrics_seg2d"):
            os.mkdir("/mnt/hdd1/zhaoxinyi/Centermask2/test_metrics_seg2d")

        print("-------------------------loading finished____________________")

        txtfile = "/mnt/hdd1/zhaoxinyi/Centermask2/test_metrics_seg2d/" + basic_model + '.txt'
        if not os.path.exists("/mnt/hdd1/zhaoxinyi/Centermask2/runs"):
            os.mkdir("/mnt/hdd1/zhaoxinyi/Centermask2/runs")
        writer = SummaryWriter(log_dir='/mnt/hdd1/zhaoxinyi/Centermask2/runs/vanilla_seg_seg2d/' + basic_model)
        count = 0

    with open(txtfile, "a+") as f:

        for epoch in trange(1, num_epochs + 1):
            LOSS = 0.
            step = 0
            test_dice = AverageMeter('dice')

            optimizer.param_groups[0]['lr'] = initial_lr * (1 - epoch / num_epochs) ** 0.9
            train_bar = tqdm(train_dataloader)

            for packs in train_bar:
                count += 1
                step += 1
                network.train()
                inputs = packs[0]
                true_masks = packs[1]

                true_masks = true_masks.to(device)
                true_masks2 = torch.cat((true_masks, 1 - true_masks), 1).to(device)
                inputs = inputs.to(device)
                masks_pred = network(inputs)

                optimizer.zero_grad()
                Dice_loss = Dice_criterion(masks_pred, true_masks2)
                Loss = Dice_loss * loss_factor['Dice']

                Loss.backward()
                optimizer.step()
                LOSS += Loss.item()
                writer.add_scalar('scalar/Loss', Loss.item(), count)
            print(basic_model + '_' + ' [epoch:%d] Loss: %.4f ' % (epoch, LOSS / (step)))

            if epoch % 1 == 0 and epoch > 0:
                print("Waiting testing")
                with torch.no_grad():
                    cnt = 0
                    for data in test_dataloader:
                        images = data[0].cuda(device)
                        labels = data[1].cuda(device)

                        cnt += 1

                        network.eval()
                        outputs = network(images)

                        predicted = torch.argmax(outputs, 1, keepdim=True)
                        mask_pred = 1 - predicted

                        mask = labels
                        pred = mask_pred
                        dice = (2.0 * torch.sum(mask * pred) + 1e-5) / (
                                torch.sum(mask) + torch.sum(pred) + 1e-5)
                        test_dice.update(dice)

                    writer.add_scalar('scalar/test_Dice', test_dice.avg)

                    if test_dice.avg > best_dice:
                        print('===========>saving best model!')
                        best_dice = test_dice.avg
                        torch.save(network.state_dict(),
                                   '/mnt/hdd1/zhaoxinyi/Centermask2/save_models_seg2d/' + basic_model + '/Best_vanilla_seg.pth')

                    print("Test:epoch=%d ,  aver_dice=%0.4f" % (epoch, test_dice.avg))
                    f.write("Test:epoch=%d , aver_dice=%0.4f" % (epoch, test_dice.avg))
                    f.write(" , best_dice=%0.4f" % (best_dice))
                    f.write('\n')
                    f.flush()
            if epoch % 10 == 0:
                torch.save(network.state_dict(),
                           '/mnt/hdd1/zhaoxinyi/Centermask2/save_models_seg2d/' + basic_model + '/' + str(epoch) + '_vanilla_seg.pth')

        f.close()
        writer.close()


if __name__ == "__main__":
    seed = 119
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--name', type=str, default='seg')

    args = parser.parse_args()

    train(args)
