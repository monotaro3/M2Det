import os
import cv2
import numpy as np
import time
from torch.multiprocessing import Pool
from utils.nms_wrapper import nms
from utils.timer import Timer
from configs.CC import Config
import argparse
from layers.functions import Detect, PriorBox
from m2det import build_net
from data import BaseTransform
from utils.core import *
from utils.pycocotools.coco import COCO
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='M2Det Testing')
parser.add_argument('-c', '--config', default='configs/m2det320_vgg.py', type=str)
parser.add_argument('-f', '--directory', default='imgs/', help='the path to demo images')
parser.add_argument('-r', '--result', default='result/', help='the path to result outputs')
parser.add_argument('-m', '--trained_model', default=None, type=str, help='Trained state_dict file path to open')
parser.add_argument('--cam', default=-1, type=int, help='camera device id')
parser.add_argument('--show', action='store_true', help='Whether to display the images')
parser.add_argument('--split', action='store_true', help='whether to split the images for test')

args = parser.parse_args()

print_info(' ----------------------------------------------------------------------\n'
           '|                       M2Det Demo Program                             |\n'
           ' ----------------------------------------------------------------------', ['yellow','bold'])

global cfg
cfg = Config.fromfile(args.config)
anchor_config = anchors(cfg)
print_info('The Anchor info: \n{}'.format(anchor_config))
priorbox = PriorBox(anchor_config)
net = build_net('test',
                size = cfg.model.input_size,
                config = cfg.model.m2det_config)
init_net(net, cfg, args.trained_model)
print_info('===> Finished constructing and loading model',['yellow','bold'])
net.eval()
with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        net = net.cuda()
        priors = priors.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
_preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)

def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
base = int(np.ceil(pow(cfg.model.m2det_config.num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(cfg.model.m2det_config.num_classes)]
cats = [_.strip().split(',')[-1] for _ in open('data/coco_labels.txt','r').readlines()]
labels = tuple(['__background__'] + cats)

def draw_detection(im, bboxes, scores, cls_inds, fps, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                    0, 1e-3 * h, colors[cls_indx], thick // 3)
        if fps >= 0:
            cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15), 0, 2e-3 * h, (255, 255, 255), thick // 2)

    return imgcv

im_path = args.directory
cam = args.cam
if cam >= 0:
    capture = cv2.VideoCapture(cam)
im_fnames = sorted((fname for fname in os.listdir(im_path) if os.path.splitext(fname)[-1] in ('.jpg','.png','.tif')))

#debug
im_fnames = [im_fnames[0]]

im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)

im_iter = iter(im_fnames)
filenum = 0

if not os.path.isdir(args.result):
    os.makedirs(args.result)

while True:
    if cam < 0:
        try:
            fname = next(im_iter)
        except StopIteration:
            break
        if 'm2det' in fname: continue # ignore the detected images
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
    else:
        ret, image = capture.read()
        if not ret:
            cv2.destroyAllWindows()
            capture.release()
            break
    loop_start = time.time()
    W, H = image.shape[1], image.shape[0]
    #debug
    print("original image:{}\nshape:{}".format(fname,image.shape))


    if not args.split:
        img = _preprocess(image).unsqueeze(0)

    if args.split:
        margin = 50
        size = cfg.model.input_size
        import math

        stride = size - margin
        H_slot = math.ceil((H - margin) / stride)
        W_slot = math.ceil((W - margin) / stride)

        #debug
        # H_slot = 1
        # W_slot = 1

        #debug
        print('W,H:{}'.format((W,H)))

        boxes =[]
        scores = []
        cls_inds = []

        #debug
        cutout_num = 0

        for h in range(H_slot):
            offset_H = stride * h if h < H_slot-1 else H - size
            for w in range(W_slot):
                offset_W = stride * w if w < W_slot-1 else W - size
                #debug
                print("offset w, h :{}".format((offset_W,offset_H)))
                cutout = image[offset_H:offset_H + size, offset_W:offset_W + size,:]
                #debug
                cutout_num += 1
                cv2.imwrite(
                    os.path.join(args.result, '{}_cutout_{}.jpg'.format(os.path.splitext(os.path.split(fname)[1])[0],cutout_num)),
                    cutout)
                w_c, h_c = cutout.shape[1], cutout.shape[0]
                # debug
                print('W_c,H_c:{}'.format((w_c, h_c)))
                cutout = _preprocess(cutout).unsqueeze(0)
                if cfg.test_cfg.cuda:
                    cutout = cutout.cuda()
                scale = torch.Tensor([w_c, h_c, w_c, h_c])
                out = net(cutout)
                box, score = detector.forward(out, priors)
                box = (box[0] * scale).cpu().numpy()
                score = score[0].cpu().numpy()

                allboxes = []
                for j in range(1, cfg.model.m2det_config.num_classes):
                    inds = np.where(score[:, j] > cfg.test_cfg.score_threshold)[0]
                    if len(inds) == 0:
                        continue
                    c_bboxes = box[inds]
                    c_scores = score[inds, j]
                    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
                    soft_nms = cfg.test_cfg.soft_nms
                    keep = nms(c_dets, cfg.test_cfg.iou,
                               force_cpu=soft_nms)  # min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
                    keep = keep[:cfg.test_cfg.keep_per_class]
                    c_dets = c_dets[keep, :]
                    allboxes.extend([_.tolist() + [j] for _ in c_dets])

                allboxes = np.array(allboxes)
                box = allboxes[:, :4]
                score = allboxes[:, 4]
                cls_ind = allboxes[:, 5]

                box[:,(0, 2)] += offset_W
                box[:,(1, 3)] += offset_H

                #debug
                print("box")
                print("shape:{}".format(box.shape))
                print("len:{}".format(len(box)))
                print(box)
                print("score")
                print("len:{}".format(len(score)))
                print("shape:{}".format(score.shape))
                print(score)

                boxes.append(box)
                scores.append(score)
                cls_inds.append(cls_ind)

        boxes = np.vstack(boxes)
        scores = np.hstack(scores)
        cls_inds = np.hstack(cls_inds)

    else:
        if cfg.test_cfg.cuda:
            img = img.cuda()
        scale = torch.Tensor([W, H, W, H])
        out = net(img)
        boxes, scores = detector.forward(out, priors)
        boxes = (boxes[0]*scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
    #
    # #debug
    # print("boxes")
    # print(boxes)
    # print("scores")
    # print(scores)

    # debug
    print("merged boxes")
    print(boxes)
    print("shape:{}".format(boxes.shape))
    print("len:{}".format(len(boxes)))
    print("merged scores")
    print(scores)
    print("len:{}".format(len(scores)))
    print("shape:{}".format(scores.shape))


    if not args.split:
        allboxes = []
        for j in range(1, cfg.model.m2det_config.num_classes):
            inds = np.where(scores[:,j] > cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            soft_nms = cfg.test_cfg.soft_nms
            keep = nms(c_dets, cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
            keep = keep[:cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allboxes.extend([_.tolist()+[j] for _ in c_dets])


        allboxes = np.array(allboxes)
        boxes = allboxes[:,:4]
        scores = allboxes[:,4]
        cls_inds = allboxes[:,5]

    loop_time = time.time() - loop_start
    #debug
    # boxes = boxes[0:2]
    # scores = scores[0:2]
    # cls_inds = cls_inds[0:2]
    print("boxes after nms:{}".format(boxes.shape))
    print("scores after nms:{}".format(scores.shape))

    print('\n'.join(['pos:{}, ids:{}, score:{:.3f}'.format('(%.1f,%.1f,%.1f,%.1f)' % (o[0],o[1],o[2],o[3]) \
            ,labels[int(oo)],ooo) for o,oo,ooo in zip(boxes,cls_inds,scores)]))
    fps = 1.0 / float(loop_time) if cam >= 0 else -1
    im2show = draw_detection(image, boxes, scores, cls_inds, fps)
    # print bbox_pred.shape, iou_pred.shape, prob_pred.shape

    if im2show.shape[0] > 1100:
        im2show = cv2.resize(im2show,
                             (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))
    # resultdir = "result"
    # if not os.path.isdir(resultdir):
    #     os.makedirs(resultdir)
    if args.show:
        # cv2.imwrite(os.path.join(resultdir,"{}.jpg".format(filenum)),im2show)
        cv2.imshow('test', im2show)
        if cam < 0:
            cv2.waitKey(5000)
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                capture.release()
                break
        plt.imshow(im2show)
    if cam < 0:
        # cv2.imwrite('{}_m2det.jpg'.format(fname.split('.')[0]), im2show)
        cv2.imwrite(os.path.join(args.result,'{}_m2det.jpg'.format(os.path.splitext(os.path.split(fname)[1])[0])), im2show)
