import torch.utils.data as data
import numpy as np
import torch

import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None

import platform
from torchvision import transforms
sysstr = platform.system()
from torchvision.transforms.functional import InterpolationMode

def dataload_tr_ts(img_path, ctp_path, case_name):

    img_list_idx = img_path + '/' + case_name
    ctp_list_idx = ctp_path + '/' + case_name

    output_image, output_mask, output_bbox, output_class, output_area = dataload_det(img_list_idx)
    output_image4 = output_image[:4, ...]
    output_image3 = output_image[-3:, ...]

    image_ctp = dataload_det_ctp(ctp_list_idx)

    image_0 = dataload_det_img0(img_list_idx)
    ret = {'image_7': output_image, 'mask': output_mask, 'bbox': output_bbox, 'class': output_class, 'area': output_area,
           'image_ctp': image_ctp, 'image_0': image_0, 'image_4': output_image4, 'image_3': output_image3,
           'image_41': output_image4[0:1, ...], 'image_42': output_image4[1:2, ...],
           'image_43': output_image4[2:3, ...], 'image_44': output_image4[3:4, ...]}

    return ret

def dataload_det(img_path):
    mask = np.load(img_path + '/' + 'mask.npy')
    inp_8 = np.load(img_path + '/' + 'img8.npy')
    inp_8 = inp_8[1:8]


    bbox = mask_bbox2d(mask)
    #
    h, w = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
    area = h * w

    output_image = inp_8.astype(np.float32)
    output_mask = mask.astype(np.float32)
    output_bbox = bbox
    output_class = 0
    output_area = area
    output_mask = np.expand_dims(output_mask, axis=0)

    return output_image, output_mask, output_bbox, output_class, output_area

def mask_bbox2d(mask):
    bbox = np.zeros(4)

    img_0 = mask[:, :].astype(int)

    coor = np.nonzero(img_0)
    coor[0].sort()
    hmin = coor[0][0]
    hmax = coor[0][-1]
    coor[1].sort()
    wmin = coor[1][0]
    wmax = coor[1][-1]

    bbox[0] = hmin.astype(int)
    bbox[2] = hmax.astype(int)

    bbox[1] = wmin.astype(int)
    bbox[3] = wmax.astype(int)


    return bbox

def dataload_det_ctp(img_path):
    inp_8 = np.load(img_path + '/' + 'ctp.npy')
    inp_8 = inp_8.transpose(2,0,1)
    image_ctp = inp_8.astype(np.float32)
    image_ctp = np.expand_dims(image_ctp, axis=0)
    return image_ctp

def dataload_det_img0(img_path):
    inp_8 = np.load(img_path + '/' + 'img8.npy')
    inp_8 = inp_8[0: 1]

    output_image = inp_8.astype(np.float32)

    return output_image

class DataLoader_CTPPC(data.Dataset):
    def __init__(self, datalist_img, path_img, path_ctp):
        super(DataLoader_CTPPC, self).__init__()

        self.file_lists = datalist_img
        self.path_img = path_img
        self.path_ctp = path_ctp

    def __len__(self):
        return len(self.file_lists)

    def __getitem__(self, index):

        case_name = self.file_lists[index].replace(self.path_img, "")

        ret = dataload_tr_ts(self.path_img, self.path_ctp, case_name)

        return ret


def dataload_seg_bbox(path):
    inp_8 = np.load(path + '/' + 'img8.npy')
    seg1 = np.load(path + '/mask.npy')

    bbox = mask_bbox2d(seg1)
    cx, cy = int((bbox[2] + bbox[0]) / 2), int((bbox[3] + bbox[1]) / 2)
    h, w = int((bbox[2] - bbox[0] + 1)), int((bbox[3] - bbox[1] + 1))

    image_crop = inp_8[:, max(0, int(cx - h / 2)): min(255, int(cx + h / 2) + 1),
                 max(0, int(cy - w / 2)): min(255, int(cy + w / 2) + 1)]
    label_crop = seg1[max(0, int(cx - h / 2)): min(255, int(cx + h / 2) + 1),
                 max(0, int(cy - w / 2)): min(255, int(cy + w / 2) + 1)]
    #
    resize = transforms.Resize([128, 128], interpolation=InterpolationMode.NEAREST)
    img_128 = np.zeros((8, 128, 128))
    for i in range(8):
        img1_1 = image_crop[i]
        img2 = Image.fromarray(img1_1)
        img_resize2 = resize(img2)
        img_128[i] = np.array(img_resize2)

    seg2 = Image.fromarray(label_crop)
    seg_resize2 = resize(seg2)
    seg_128 = np.array(seg_resize2)
    seg_128 = np.expand_dims(seg_128, axis=0)

    return img_128, seg_128


class DataLoader_seg(data.Dataset):
    def __init__(self, datapath):
        super(DataLoader_seg, self).__init__()

        self.file_lists = datapath

    def __len__(self):
        return len(self.file_lists)

    def __getitem__(self, index):
        img, mask = dataload_seg_bbox(self.file_lists[index])

        return torch.from_numpy(np.array(img)).float(), torch.from_numpy(np.array(mask)).long()

def dataload_ts(path):

    inp_8 = np.load(path + '/' + 'img8.npy')
    seg = np.load(path + '/mask.npy')
    seg = np.expand_dims(seg, axis=0)

    return inp_8, seg