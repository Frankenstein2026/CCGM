# -*- coding: utf-8 -*-
from train_det import *
from model_seg import *

def accuracy(mask, pred):
    tp = torch.sum((mask == 1) & (pred == 1)).float()
    tn = torch.sum((mask == 0) & (pred == 0)).float()
    fp = torch.sum((mask == 0) & (pred == 1)).float()
    fn = torch.sum((mask == 1) & (pred == 0)).float()
    return (tp + tn) / (tp + tn + fp + fn+1e-5)

def precision(mask, pred):
    tp = torch.sum((mask == 1) & (pred == 1)).float()
    fp = torch.sum((mask == 0) & (pred == 1)).float()
    return tp / (tp + fp+1e-5)

def recall(mask, pred):
    tp = torch.sum((mask == 1) & (pred == 1)).float()
    fn = torch.sum((mask == 1) & (pred == 0)).float()
    return tp / (tp + fn+1e-5)

def f1_score(mask, pred):
    prec = precision(mask, pred)
    rec = recall(mask, pred)
    return 2 * (prec * rec) / (prec + rec+1e-5)
from medpy.metric.binary import hd, hd95, assd
def hd95_assd(mask, pred):
    mask = mask.numpy()
    pred = pred.numpy()
    if np.sum(pred) > 0:
        h_dist95 = hd95(mask, pred)
        assd_value = assd(mask, pred)
        cnt = 1
    else:
        h_dist95 = 0
        assd_value = 0
        cnt = 0
    return h_dist95, assd_value, cnt
def resize_bbox(bbox_output):
    h, w = bbox_output[2]-bbox_output[0], bbox_output[3]-bbox_output[1]
    x, y = (bbox_output[2]+bbox_output[0])/2., (bbox_output[3]+bbox_output[1])/2.
    bbox = torch.zeros((4))
    bbox[0] = max(0, int(x - h / 2.))
    bbox[1] = max(0, int(y - w / 2.))
    bbox[2] = min(255, int(x + h / 2.))
    bbox[3] = min(255, int(y + w / 2.))
    return bbox

def test(args):
    test_path_img = '/mnt/hdd1/zhaoxinyi/DATA_new/Data2D_256_8c_random/imagesTs/'
    test_list_img = os.listdir(test_path_img)
    test_list_img.sort()

    test_path_ctp = '/mnt/hdd1/zhaoxinyi/DATA_new/ctp_prepare/random_imagesTs/'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

    model_det = get_model_det(args).cuda().to(device)
    model_det.load_state_dict(
        torch.load('/Best_vanilla_det.pth'), strict=False)

    model_seg = get_model_seg(inc=8).cuda().to(device)
    model_seg = nn.DataParallel(model_seg)
    

    model_seg.load_state_dict(
        torch.load('/Best_vanilla_seg.pth'))


    with torch.no_grad():
        pred1 = []
        pred2 = []
        pred3 = []
        pred4 = []
        pred5 = []
        pred6 = []
        pred0 = []
        cnt5 = 0
        for i in trange(len(test_list_img)):
            ret_test = dataload_tr_ts(test_path_img, test_path_ctp, test_list_img[i])
            ret_test = {key: torch.tensor(value)[None, ...] for key, value in ret_test.items()}
            inputs_test = dict_to_device(ret_test, device)

            ret_test_img_seg, ret_test_mask_seg = dataload_ts(test_path_img + '/' + test_list_img[i])
            ret_test_img_seg = torch.tensor(ret_test_img_seg)

            
            model_det.eval()            
            model_seg.eval()

            test_output_proposals, _ = model_det(inputs_test)
            test_bbox = test_output_proposals[0].pred_boxes.tensor
            if test_bbox.numel() == 0:
                pred1.append(0.)
                pred2.append(0.)
                pred3.append(0.)
                pred4.append(0.)
                pred5.append(0.)
                pred6.append(0.)
                pred0.append(0.)
            else:

                bbox_output = test_bbox[0].squeeze(0)
                bbox = resize_bbox(bbox_output)

                cropped = ret_test_img_seg[:, int(bbox[0]): int(bbox[2]) + 1, int(bbox[1]): int(bbox[3]) + 1]
                cropped = cropped.unsqueeze(0)
                resized = F.interpolate(cropped, size=(128, 128), mode='nearest')

                outputs_seg = model_seg(resized)

                predicted = torch.argmax(outputs_seg, 1, keepdim=True)
                mask_pred = 1 - predicted
                mask_pred = mask_pred.type(torch.float64)

                restored = F.interpolate(mask_pred, size=(cropped.shape[2], cropped.shape[3]), mode='nearest')

                pred = torch.zeros((256, 256))
                pred[int(bbox[0]): int(bbox[2]) + 1, int(bbox[1]): int(bbox[3]) + 1] = restored[0, 0]

                mask = ret_test['mask'].squeeze(0).cpu()

                dice = (2.0 * torch.sum(mask * pred) + 1e-5) / (
                        torch.sum(mask) + torch.sum(pred) + 1e-5)
                acc = accuracy(mask, pred)
                pre = precision(mask, pred)
                rec = recall(mask, pred)
                mask = mask.squeeze(0)
                h_dist95, assd, cnt = hd95_assd(mask, pred)

                pred1.append(float(dice.numpy()))
                pred2.append(float(acc.numpy()))
                pred3.append(float(pre.numpy()))
                pred4.append(float(rec.numpy()))
                if cnt == 1:
                    cnt5 = cnt5+1
                pred5.append(float(h_dist95))
                pred6.append(float(assd))

        print("Test avg_dice=%0.4f" % (sum(pred1)/len(pred1)))
        print("Test avg_acc=%0.4f" % (sum(pred2) / len(pred2)))
        print("Test avg_pre=%0.4f" % (sum(pred3) / len(pred3)))
        print("Test avg_recall=%0.4f" % (sum(pred4) / len(pred4)))
        print("Test avg_hd95=%0.4f" % (sum(pred5) / cnt5))
        print("Test avg_assd=%0.4f" % (sum(pred6) / cnt5))
        print("Test std_dev_dice=%0.4f" % (np.std(pred1)))
        print("Test std_dev_acc=%0.4f" % (np.std(pred2)))
        print("Test std_dev_pre=%0.4f" % (np.std(pred3)))
        print("Test std_dev_recall=%0.4f" % (np.std(pred4)))
        print("Test std_dev_hd95=%0.4f" % (np.std(pred5)))
        print("Test std_dev_assd=%0.4f" % (np.std(pred6)))

if __name__ == "__main__":
    seed = 119
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6')
    parser.add_argument('--name', type=str, default='_')
    parser.add_argument('--class_weight', type=float, default=1)
    parser.add_argument('--reg_weight', type=float, default=1)
    parser.add_argument('--ctrness_weight', type=float, default=1)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--thresh_with_ctr', type=bool, default=True)
    parser.add_argument('--thresh', type=float, default=0.1)
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
    parser.add_argument('--if_concat_cc', type=int, default=1)
    parser.add_argument('--if_mamba', type=int, default=0)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--pth_mamba', type=str, default='/mnt/ssd2/zxy/centermask2/detectron_zxy/detectron2/saved_models_cm/mamba_det_load_/Best_vanilla_det.pth')
    parser.add_argument('--pth_trans', type=str, default='/mnt/ssd2/zxy/centermask2/detectron_zxy/detectron2_gyc/saved_models_cm/mamba_det_loadconcat_cc_nomamba/Best_vanilla_det.pth')
    parser.add_argument('--fix_mamba', type=int, default=0)
    parser.add_argument('--fix_trans', type=int, default=0)

    args = parser.parse_args()
    test(args)
