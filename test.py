import torch
import os
import KGnet
import numpy as np
from dataset_kaggle import Kaggle
from dataset_plant import Plant
from dataset_neural import Neural
import argparse
import cv2
import postprocessing
import time
import nms
import colorsys
import random
import math
import csv
# from matplotlib import pyplot as plt
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageDraw

def parse_args():

    parser = argparse.ArgumentParser(description="InstanceHeat")
    parser.add_argument("--data_dir", help="data directory", default="../../../../Datasets/kaggle/", type=str)
    parser.add_argument("--resume", help="resume file", default="end_model.pth", type=str)
    parser.add_argument('--input_h', type=int, default=512, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--save_img', type=bool, default=False, help='save img or not')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms_thresh')
    parser.add_argument('--seg_thresh', type=float, default=0.5, help='seg_thresh')
    parser.add_argument("--dataset", help="training dataset", default='kaggle', type=str)
    args = parser.parse_args()
    return args

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]

class InstanceHeat(object):
    def __init__(self):
        self.model = KGnet.resnet50(pretrained=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = {'kaggle': Kaggle, 'plant': Plant, 'neural': Neural}

    def data_parallel(self):
        self.model = torch.nn.DataParallel(self.model)

    def load_weights(self, resume, dataset):
        self.model.load_state_dict(torch.load(os.path.join('weights_'+dataset, resume)))

    def map_mask_to_image(self, mask, img, color):
        # color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def show_heat_mask(self, mask):
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        return heatmap

    def imshow_kp(self, kp, img_in):
        h,w = kp.shape[2:]
        img = cv2.resize(img_in, (w, h))
        colors = [(0,0,0.9),(0.9,0,0),(0.9,0,0.9),(0.9,0.9,0), (0.2,0.9,0.9)]
        for i in range(kp.shape[1]):
            img = self.map_mask_to_image(kp[0,i,:,:], img, color=colors[i])
        return img

    def test_inference(self, args, image, bbox_flag=False):
        height, width, c = image.shape

        img_input = cv2.resize(image, (args.input_w, args.input_h))
        img_input = torch.FloatTensor(np.transpose(img_input.copy(), (2, 0, 1))).unsqueeze(0) / 255 - 0.5
        img_input = img_input.to(self.device)

        with torch.no_grad():
            begin = time.time()
            pr_c0, pr_c1, pr_c2, pr_c3, feat_seg = self.model.forward_dec(img_input)
            print("forward time is {:.4f}".format(time.time() - begin))
            pr_kp0, pr_short0, pr_mid0 = pr_c0
            pr_kp1, pr_short1, pr_mid1 = pr_c1
            pr_kp2, pr_short2, pr_mid2 = pr_c2
            pr_kp3, pr_short3, pr_mid3 = pr_c3
        if(self.device.type != 'cpu'):
            torch.cuda.synchronize()# Thai, remove when device is cpu
        skeletons0 = postprocessing.get_skeletons_and_masks(pr_kp0, pr_short0, pr_mid0)
        skeletons1 = postprocessing.get_skeletons_and_masks(pr_kp1, pr_short1, pr_mid1)
        skeletons2 = postprocessing.get_skeletons_and_masks(pr_kp2, pr_short2, pr_mid2)
        skeletons3 = postprocessing.get_skeletons_and_masks(pr_kp3, pr_short3, pr_mid3)

        skeletons0 = postprocessing.refine_skeleton(skeletons0)
        skeletons1 = postprocessing.refine_skeleton(skeletons1)
        skeletons2 = postprocessing.refine_skeleton(skeletons2)
        skeletons3 = postprocessing.refine_skeleton(skeletons3)

        bboxes = postprocessing.gather_skeleton(skeletons0, skeletons1, skeletons2, skeletons3)
        bboxes = nms.non_maximum_suppression_numpy(bboxes, nms_thresh=args.nms_thresh)
        if bbox_flag:
            return bboxes
        if bboxes is None:
            return None

        with torch.no_grad():
            predictions = self.model.forward_seg(feat_seg, [bboxes])
        predictions = self.post_processing(args, predictions, width, height)
        return predictions

    def post_processing(self, args, predictions, image_w, image_h):
        if predictions is None:
            return predictions
        out_masks = []
        out_dets = []
        mask_patches, mask_dets = predictions
        for mask_b_patches, mask_b_dets in zip(mask_patches, mask_dets):
            for mask_n_patch, mask_n_det in zip(mask_b_patches, mask_b_dets):
                mask_patch = mask_n_patch.data.cpu().numpy()
                mask_det = mask_n_det.data.cpu().numpy()
                y1, x1, y2, x2, conf = mask_det
                y1 = np.maximum(0, np.int32(np.round(y1)))
                x1 = np.maximum(0, np.int32(np.round(x1)))
                y2 = np.minimum(np.int32(np.round(y2)), args.input_h - 1)
                x2 = np.minimum(np.int32(np.round(x2)), args.input_w - 1)

                mask = np.zeros((args.input_h, args.input_w), dtype=np.float32)
                mask_patch = cv2.resize(mask_patch, (x2 - x1, y2 - y1))

                mask[y1:y2, x1:x2] = mask_patch
                mask = cv2.resize(mask, (image_w, image_h))
                mask = np.where(mask >= args.seg_thresh, 1, 0)

                y1 = float(y1) / args.input_h * image_h
                x1 = float(x1) / args.input_w * image_w
                y2 = float(y2) / args.input_h * image_h
                x2 = float(x2) / args.input_w * image_w

                out_masks.append(mask)
                out_dets.append([y1,x1,y2,x2, conf])
        return [np.asarray(out_masks, np.float32), np.asarray(out_dets, np.float32)]

    def rotate_image(self, image: Image, angle: float):
        """
        Rotate image.

        Parameters
        ----------
        image : Image
        angle : float

        Returns
        -------
        Image

        """
        return image.rotate(angle, expand=True)

    def transform_pixel_coordinates(self,
            pixel_coordinates: Tuple[int, int],
            angle: float,
            image: Image,
            rotated_image: Image,
    ) -> Tuple[int, int]:
        """
        Transform pixel coordinates.

        Parameters
        ----------
        pixel_coordinates : Tuple[int, int]
        angle : float
        image : Image
        rotated_image : Image

        Returns
        -------
        Tuple[int, int]

        """

        x, y = pixel_coordinates

        center = (image.width / 2, image.height / 2)
        transformed_center = (rotated_image.width / 2, rotated_image.height / 2)

        angle_radians = -np.deg2rad(angle)

        x -= center[0]
        y -= center[1]

        x_transformed = x * np.cos(angle_radians) - y * np.sin(angle_radians)
        y_transformed = x * np.sin(angle_radians) + y * np.cos(angle_radians)

        x_transformed += transformed_center[0]
        y_transformed += transformed_center[1]

        return int(x_transformed), int(y_transformed)

    def draw_images( self, image, rotated_image, pixel_coordinates, transformed_pixel_coordinates ):
        """
        Draw images and pixel.

        Parameters
        ----------
        image : Image
        rotated_image : Image
        pixel_coordinates : Tuple[int, int]
        transformed_pixel_coordinates : Tuple[int, int]

        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(image)
        axes[0].scatter(*pixel_coordinates, color="y", s=50)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(rotated_image)
        axes[1].scatter(*transformed_pixel_coordinates, color="y", s=50)
        axes[1].set_title("Rotated Image")
        axes[1].axis("off")
        plt.show()
    def imshow_instance_segmentation(self,
                                     masks,
                                     dets,
                                     out_img,
                                     img_id=None,
                                     save_flag=False,
                                     show_box=False,
                                     save_path=None):

        colors = random_colors(masks.shape[0])
        i=0
        cp_img = out_img.copy()
        # cp_img1 = out_img.copy()
        cp_img1 = np.ones(out_img.shape)
        cp_img1 = np.uint8(cp_img1 * out_img)
        vis_img_line = out_img.copy()
        out_img_E = out_img.copy()
        cntsElps = []

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        colorText = (0, 0, 255)
        thickness = 1

        for mask, det, color in zip(masks, dets, colors):
            i = i + 1

            # color = np.random.rand(3)
            # m_img = mask.copy()
            for c in range(3):
                cp_img[:, :, c] = np.where(mask == 1,
                                          255,
                                          cp_img[:, :, 2] * 0)
            predicted_area = mask.sum()
            gray = cv2.cvtColor(cp_img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 150, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            if(len(cnt) < 5):
                continue
            ellipse = cv2.fitEllipse(cnt)
            (xc, yc), (width, height), angle = ellipse
            rminor = min(width, height) / 2
            xtop = xc + math.cos(math.radians(angle)) * rminor
            ytop = yc + math.sin(math.radians(angle)) * rminor
            cv2.line(out_img_E, (int(xtop), int(ytop)), (int(xc), int(yc)), (0, 0, 255), 1)

            cntsElps.append(ellipse)
            cv2.putText(out_img_E, str(i), (int(ellipse[0][0]), int(ellipse[0][1])), font, fontScale,
                        colorText, thickness, cv2.LINE_AA)

            print("Ellipse nb: " + str(i) + " has angle: " + str(ellipse[2]) + "\n")
            print("Ellipse nb: " + str(i) + " has center x: " + str(ellipse[0][0]) + "\n")
            print("Ellipse nb: " + str(i) + " has center y: " + str(ellipse[0][1]) + "\n")
            print("Ellipse nb: " + str(i) + " has minor: " + str(ellipse[1][0]) + "\n")
            print("Ellipse nb: " + str(i) + " has major: " + str(ellipse[1][1]) + "\n")
            print("Mask predicted_area: " + str(i) + " : " + str(predicted_area) + "\n")
            # cv2.ellipse(out_img_E, ellipse, (255, 255, 255), 1)
            cv2.ellipse(out_img_E, ellipse, (color[0] * 255, color[1] * 255, color[2] * 255), 1)
            mask_I = np.zeros((gray.shape[0], gray.shape[1]))


            vis_cell = np.zeros((gray.shape[0], gray.shape[1], 3))
            vis_cell_line = np.zeros((gray.shape[0], gray.shape[1], 3))
            NUMLAYER = 4
            for l in range(0, NUMLAYER + 1):
                new_ellipse = (ellipse[0], (ellipse[1][0] * (l + 1) / NUMLAYER, ellipse[1][1] * (l + 1) / NUMLAYER), ellipse[2])

                cv2.ellipse(vis_cell, new_ellipse, (color[0] * 255, color[1] * 255, color[2] * 255), -1)
                cv2.ellipse(vis_cell_line, new_ellipse, (color[0] * 255, color[1] * 255, color[2] * 255), 1)
                cv2.ellipse(vis_img_line, new_ellipse, (color[0] * 255, color[1] * 255, color[2] * 255), 1)
                mask_I = np.all(vis_cell == [color[0] * 255, color[1] * 255, color[2] * 255], axis=-1)
                mask_I_line = np.all(vis_cell_line == [color[0] * 255, color[1] * 255, color[2] * 255], axis=-1)

                total_intense = (mask_I * out_img[:, :, 1]).sum()
                count_eclipse = (mask_I == True).sum()
                print(l, ": total_intense: ", total_intense)
                print(l, ": count_eclipse: ", count_eclipse)
                print(l, ": average intense: ", total_intense/count_eclipse)
                # cv2.imshow("vis_cell", out_img[:, :, 1])
                # cv2.imshow("mask_I", mask_I.astype(np.uint8) *  255)
                # cv2.imshow("vis_img_line", vis_img_line)
                # cv2.imshow("out_img", out_img[:, :, 1])
                # cv2.waitKey(0)

            #rotate the image base on the mask
            angle_ = angle - 90
            img = out_img.copy()
            h, w, _ = img.shape
            pixel_coordinates_ = (w/2, h/2)

            translation_matrix = np.array([
                [1, 0, w/2 - xc],
                [0, 1, h/2 - yc]
            ], dtype=np.float32)

            # (xc, yc), (width, height), angle = ellipse
            translated_image = cv2.warpAffine(src=img, M=translation_matrix, dsize=(w, h))
            rot_mat = cv2.getRotationMatrix2D(pixel_coordinates_, angle_, 1)
            rotated_img = cv2.warpAffine(translated_image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
            rotated_img_after = rotated_img.copy()
            new_ellipse = (
            (int(w/2), int(h/2)), (int(ellipse[1][0] * (NUMLAYER + 1) / NUMLAYER), int(ellipse[1][1] * (NUMLAYER + 1) / NUMLAYER)), 90)
            vis_cell = np.zeros((gray.shape[0], gray.shape[1], 3))
            cv2.ellipse(vis_cell, new_ellipse, (color[0] * 255, color[1] * 255, color[2] * 255), -1)
            mask_I = np.all(vis_cell == [color[0] * 255, color[1] * 255, color[2] * 255], axis=-1)
            vis_cell_right = vis_cell.copy()
            vis_cell_bot = vis_cell.copy()

            vis_cell_right[0:h, 0:int(w/2)] = (0, 0, 0)
            vis_cell_bot[0:int(h/2), 0:w] = (0, 0, 0)
            mask_I_right = np.all(vis_cell_right == [color[0] * 255, color[1] * 255, color[2] * 255], axis=-1)
            mask_I_bot = np.all(vis_cell_bot == [color[0] * 255, color[1] * 255, color[2] * 255], axis=-1)


            total_intense = (mask_I * rotated_img[:, :, 1]).sum()
            total_intense_right = (mask_I_right * rotated_img[:, :, 1]).sum()
            total_intense_bot = (mask_I_bot * rotated_img[:, :, 1]).sum()
            if(total_intense_right < total_intense - total_intense_right):#if total intense on right smaller than total intense on the left
                rotated_img_after = cv2.flip(rotated_img_after, 1)
            if (
                    total_intense_bot < total_intense - total_intense_bot):  # if total intense on right smaller than total intense on the left
                rotated_img_after = cv2.flip(rotated_img_after, 0)
            print(str(total_intense), "-----" ,str(total_intense_right), "------------------", str(total_intense - total_intense_right))
            print(str(total_intense), "-----" ,str(total_intense_bot), "------------------", str(total_intense - total_intense_bot))
            NUMCIRCLEPART = 8
            print("--------------CELL ", str(i), " --------------")
            for l in range(0, NUMCIRCLEPART):
                vis_cell_segment = np.zeros((gray.shape[0], gray.shape[1], 3))

                center = new_ellipse[0]
                axes = new_ellipse[1]
                angle = new_ellipse[2]
                startAngle = 45 * l
                endAngle = 45 * (l + 1)

                cv2.ellipse(vis_cell_segment, center, axes, angle, startAngle, endAngle, (color[0] * 255, color[1] * 255, color[2] * 255), -1)
                mask_I_segment = np.all(vis_cell_segment == [color[0] * 255, color[1] * 255, color[2] * 255], axis=-1)
                total_intense_segment = (mask_I_segment * rotated_img[:, :, 1]).sum()
                count_eclipse_segment = (mask_I_segment == True).sum()
                print(l, ": total_intense_segment: ", total_intense_segment)
                print(l, ": count_eclipse_segment: ", count_eclipse_segment)
                print(l, ": average intense segment: ", total_intense_segment / count_eclipse_segment)

                # cv2.imshow("vis_cell_segment", vis_cell_segment)
                # cv2.waitKey(0)
                # count_eclipse = (mask_I == True).sum()

            # cv2.imshow("rotated_img", rotated_img)
            # cv2.imshow("rotated_img_after", rotated_img_after)
            # cv2.waitKey(0)



            if show_box:
                y1,x1,y2,x2,conf = det
                cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], 1, 1)
                cv2.putText(out_img, "{:.4f}".format(conf), (int(x1),int(y1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, 1)
            maskRev = 1 - mask
            mask1 = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            maskRev = np.repeat(maskRev[:, :, np.newaxis], 3, axis=2)
            # mskd = cp_img1 * mask
            clmsk = np.ones(mask1.shape) * mask1
            clmsk[:, :, 0] = clmsk[:, :, 0] * (color[0] * 255)
            clmsk[:, :, 1] = clmsk[:, :, 1] * (color[1] * 255)
            clmsk[:, :, 2] = clmsk[:, :, 2] * (color[2] * 255)
            # cv2.imshow("clmsk", clmsk)
            # cv2.imshow("cp_img1", cp_img1)
            # cv2.waitKey(0)
            cp_img1 = np.uint8(cp_img1 * np.uint8(maskRev) + clmsk)
            # apply_mask(image=cp_img1, mask=mask, color=color, alpha=0.8)
            cv2.imshow("clmsk", clmsk)
            cv2.imshow("cp_img1", cp_img1)
            cv2.waitKey(0)
        #Get green channel

        # cv2.imshow("cp_img", cp_img)
        # cv2.waitKey(0)
        green_img = out_img.copy()
        gray_ori = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Green channel", green_img)
        cv2.imshow("out_img_E", out_img_E)
        # cv2.imshow("gray", gray_ori)
        # cv2.waitKey(0)

        cv2.waitKey(0)
        #For each eclipse

        mask_E = np.zeros((out_img_E.shape[0], out_img_E.shape[1], 1), dtype="uint8")
        for e in cntsElps:

            cv2.ellipse(green_img, e, (255, 255, 255), 1)
            cv2.ellipse(mask_E, e, (255, 255, 255), 1)
        cv2.imshow("mask_E", mask_E)
        cv2.waitKey(0)
        #Get eclipse bigger to cover all the cell
        #get the average intensitive
        # make new smaller eclipse
        #get average intensive color of smaller eclipse

        #save intensitive of smaller eclipse

        #

        if save_flag:
            try:
                cv2.imwrite(os.path.join(save_path, img_id+".png"), np.uint8(out_img))
                cv2.imwrite(os.path.join(save_path, img_id+"_eclipse.png"), np.uint8(cp_img))
                cv2.imwrite(os.path.join(save_path, img_id+"_Eclipse1.png"), np.uint8(out_img_E))
            except Exception as e:
                print("Oops!  Error: ", e)
        # cv2.imshow('out_img', np.uint8(out_img))
        # k = cv2.waitKey(0)
        # if k & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit(1)

    def test(self, args):
        self.load_weights(resume=args.resume, dataset=args.dataset)
        self.model = self.model.to(self.device)
        self.model.eval()

        if args.save_img:
            save_path = 'save_result_'+args.dataset
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        else:
            save_path = None
        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir, phase='test')
        all_time = []
        for index in range(len(dsets)):
            time_begin = time.time()
            img = dsets.load_image(index)

            predictions = self.test_inference(args, img)

            if predictions is None:
                continue
            mask_patches, mask_dets = predictions
            all_time.append(time.time()-time_begin)
            self.imshow_instance_segmentation(mask_patches, mask_dets,
                                              out_img=img.copy(),
                                              img_id=dsets.img_ids[index],
                                              save_flag= args.save_img,
                                              save_path=save_path)
        all_time = all_time[1:]
        print('avg time is {}'.format(np.mean(all_time)))
        print('FPS is {}'.format(1./np.mean(all_time)))


if __name__ == '__main__':
    args = parse_args()
    object_is = InstanceHeat()
    object_is.test(args)
