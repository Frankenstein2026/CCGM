###########################
import argparse
from tensorboardX import SummaryWriter

from model_det import *
from torch import optim
from tqdm import tqdm, trange
from torch.utils.data.dataloader import *
from dataset import *

def dict_to_device(d, device):
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            d[key] = value.to(device)
    return d


def calculate_iou_2d(bbox1, bbox2):

    # 计算交集的坐标
    inter_x1 = torch.max(bbox1[:, 0][:, None], bbox2[:, 0])
    inter_y1 = torch.max(bbox1[:, 1][:, None], bbox2[:, 1])
    inter_x2 = torch.min(bbox1[:, 2][:, None], bbox2[:, 2])
    inter_y2 = torch.min(bbox1[:, 3][:, None], bbox2[:, 3])

    # 计算交集面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算 bbox1 和 bbox2 的面积
    bbox1_area = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    bbox2_area = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])

    # 计算并返回 IoU
    iou = inter_area / (bbox1_area[:, None] + bbox2_area - inter_area)
    return iou

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


def get_model_det(args):
    model = Mamba_cross(args)
    return model

def train(args):
    basic_model = "mamba_det_load" + args.name

    torch.autograd.set_detect_anomaly(True)
    patients_tr_img = '/mnt/hdd1/zhaoxinyi/DATA_new/Data2D_256_8c_random/imagesTr/'
    torch.autograd.set_detect_anomaly(True)
    patients_tr_ctp = '/mnt/hdd1/zhaoxinyi/DATA_new/ctp_prepare/random_imagesTr/'
    tr_list = [os.path.join(patients_tr_img, file_name) for file_name in os.listdir(patients_tr_img)]

    patients_ts_img = '/mnt/hdd1/zhaoxinyi/DATA_new/Data2D_256_8c_random/imagesTs/'
    patients_ts_ctp = '/mnt/hdd1/zhaoxinyi/DATA_new/ctp_prepare/random_imagesTs/'
    test_list = os.listdir(patients_ts_img)
    test_list.sort()

    dataset_tr = DataLoader_CTPPC(tr_list, patients_tr_img, patients_tr_ctp)
    batch_size = args.bs
    data_loader = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=4)


    print("-------------------------loading finished____________________")

    num_epochs = 2000

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

    model = get_model_det(args)
    model_keys = model.state_dict()

    pth_trans, pth_mamba = {}, {}
    if args.pth_trans != '':
        pth_trans = torch.load(args.pth_trans)
    if args.pth_mamba != '':
        pth_mamba = torch.load(args.pth_mamba)

    if args.load:
        for key in model_keys:
            if 'mamba' in key and key in pth_mamba.keys():
                model_keys[key] = pth_mamba[key]
            if 'mamba' not in key and key in pth_trans.keys():
                model_keys[key] = pth_trans[key]
        model.load_state_dict(model_keys)
        print('-----------------load pth success----------------')


    if torch.cuda.is_available():
        model.cuda().to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          args.initial_lr, weight_decay=args.weight_decay,
                          momentum=0.99, nesterov=True)


    if not os.path.exists("./saved_models/" + basic_model):
        if not os.path.exists("./saved_models"):
            os.mkdir("./saved_models")
        os.mkdir("./saved_models/" + basic_model)

    if not os.path.exists("./test_metrics"):
        os.mkdir("./test_metrics")
    txtfile = "./test_metrics/" + basic_model + '.txt'
    if not os.path.exists("./runs"):
        os.mkdir("./runs")
    writer = SummaryWriter(log_dir='runs/det/' + basic_model)
    count = 0
    best_iou = 0
    loss_factor = {'class': args.class_weight, 'reg': args.reg_weight, 'ctrness': args.ctrness_weight}



    with open(txtfile, "a+") as f:

        for epoch in trange(1, num_epochs + 1):
            LOSS = 0.
            step = 0

            optimizer.param_groups[0]['lr'] = args.initial_lr * (1 - epoch / num_epochs) ** 0.9
            train_bar = tqdm(data_loader)

            for packs in train_bar:
                count += 1
                step += 1
                model.train()

                packs = dict_to_device(packs, device)
                output_proposals, output_loss = model(packs)

                Loss = loss_factor['class'] * output_loss['loss_fcos_cls'] \
                       + loss_factor['reg'] * output_loss['loss_fcos_loc'] \
                       + loss_factor['ctrness'] * output_loss['loss_fcos_ctr']
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()
                LOSS += Loss.item()

                writer.add_scalar('scalar/Loss', Loss.item(), count)
                writer.add_scalar('scalar/loss_fcos_cls', output_loss['loss_fcos_cls'].item(), count)
                writer.add_scalar('scalar/loss_fcos_loc', output_loss['loss_fcos_loc'].item(), count)
                writer.add_scalar('scalar/loss_fcos_ctr', output_loss['loss_fcos_ctr'].item(), count)

            print(basic_model + '_' + ' [epoch:%d] Loss: %.4f ' % (epoch, LOSS / (step)))
            iou0 = AverageMeter('iou0')

            if epoch % 5 == 0 and epoch > 0:
                print("Waiting testing")
                with torch.no_grad():
                    for i in trange(len(test_list)):
                        ret_test = dataload_tr_ts(patients_ts_img, patients_ts_ctp,test_list[i])
                        ret_test = {key: torch.tensor(value)[None, ...] for key, value in ret_test.items()}
                        inputs_test = dict_to_device(ret_test, device)
                        model.eval()
                        test_output_proposals, _ = model(inputs_test)

                        test_bbox = test_output_proposals[0].pred_boxes.tensor

                        test_iou4 = []
                        for bbox in test_bbox:
                            b = bbox.unsqueeze(0)
                            test_iou = calculate_iou_2d(ret_test['bbox'], b)
                            test_iou4.append(test_iou.squeeze(0).cpu().numpy())

                        if test_bbox.numel() == 0:
                            iou0.update(np.array(0))
                        else:
                            iou0.update(test_iou4[0])

                    writer.add_scalar('scalar/iou0', iou0.avg.item(), epoch)
                    if iou0.count == len(test_list):
                        if iou0.avg > best_iou:
                            print('===========>saving best model!')
                            best_iou = iou0.avg
                            torch.save(model.state_dict(),
                                       './saved_models/' + basic_model + '/Best_vanilla_det.pth')

                        print("Test:epoch=%d ,  aver_iou=%0.4f" % (epoch, iou0.avg.item()))
                        f.write("Test:epoch=%d , aver_iou=%0.4f" % (epoch, iou0.avg.item()))


                    f.write('\n')
                    f.flush()
            if epoch % 1000 == 0 and epoch > 1:
                torch.save(model.state_dict(),
                               './saved_models/' + basic_model + '/' + str(epoch) + '_vanilla_det.pth')

        f.close()
        writer.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6')
    parser.add_argument('--name', type=str, default='_')
    parser.add_argument('--class_weight', type=float, default=1)
    parser.add_argument('--reg_weight', type=float, default=1)
    parser.add_argument('--ctrness_weight', type=float, default=1)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--thresh_with_ctr', type=bool, default=True)
    parser.add_argument('--thresh', type=float, default=0.1)
    parser.add_argument('--initial_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=3e-5)
    parser.add_argument('--radius', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--seed', type=int, default=1301)
    parser.add_argument('--iou', type=str, default='iou')
    parser.add_argument('--backbone', type=str, default="V-99-eSE")

    parser.add_argument('--fpn_inc', type=int, default=512)
    parser.add_argument('--vov_outc', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--m', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--ctp_c', type=int, default=16)
    parser.add_argument('--num_c', type=int, default=1)
    parser.add_argument('--if_concat_cc', type=int, default=0)
    parser.add_argument('--if_mamba', type=int, default=0)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--pth_mamba', type=str, default='/saved_models/mamba_det_load_/Best_vanilla_det.pth')
    parser.add_argument('--pth_trans', type=str, default='/saved_models/mamba_det_loadconcat_cc_nomamba/Best_vanilla_det.pth')
    parser.add_argument('--fix_mamba', type=int, default=0)
    parser.add_argument('--fix_trans', type=int, default=0)


    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train(args)

