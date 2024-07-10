import torch
from SPPE.src.main_fast_inference import InferenNet_fastRes50
from SPPE.src.utils.img import crop_dets
from pPose_nms import pose_nms
from SPPE.src.utils.eval import getPrediction
from Lhrnet.lhrnet import LiteHRNet


class SPPE_FastPose(object):
    def __init__(self,
                 pose_model,
                 input_height=384,
                 input_width=288,
                 device='cuda'):
        assert pose_model in ['hrnet', 'alpha-pose'], '{} model is not support yet!'.format(pose_model)
        print('骨架提取的模型是：{}'.format(pose_model))
        self.inp_h = input_height
        self.inp_w = input_width
        self.device = device
        if pose_model == 'alpha-pose':
            self.model = InferenNet_fastRes50().to(device)
        else:
            extra = dict(
                stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
                num_stages=3,
                stages_spec=dict(
                    num_modules=(3, 8, 3),
                    num_branches=(2, 3, 4),
                    num_blocks=(2, 2, 2),
                    module_type=('LITE', 'LITE', 'LITE'),
                    with_fuse=(True, True, True),
                    reduce_ratios=(8, 8, 8),
                    num_channels=(
                        (40, 80),
                        (40, 80, 160),
                        (40, 80, 160, 320),
                    )),)
            self.model = self.model = LiteHRNet(extra, in_channels=3).to(device)
            checkpoint = torch.load(r'J:\youda\Lhrnet\0517.pt', map_location=self.device)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, image, bboxs, bboxs_scores):
        # 根据框的大小裁剪图
        # 在一开始读取图片的时候已经转换为了BGR格式，在经过crop_det后图片进行了像素归一化以及减去了平均值
        # print('len:', len(bboxs))
        inps, pt1, pt2 = crop_dets(image, bboxs, self.inp_h, self.inp_w)
        # inps = inps[0].numpy()
        # inps_2 = np.transpose(inps, (1, 2, 0))*255
        # cv2.imshow('ss', inps_2)
        # cv2.waitKey(0)

        # inps_1 = inps[0].numpy()
        # inps_1 = np.transpose(inps_1, (1, 2, 0))
        # np.savetxt('task.txt', inps_1[:, :, 0])
        # cv2.imwrite("demo_2.jpg", inps_1)
        # cv2.imshow("sss", inps_1)
        #
        # cv2.waitKey(100000)
        # if len(inps) == 2:
        #     inps_2 = inps[1].numpy()
        #     inps_2 = np.transpose(inps_2, (1, 2, 0))
        #     print("pt1.shape:", inps_2.shape)
        #     cv2.imshow("第二章", inps_2)
        #     cv2.waitKey(1000)

        # 将裁剪出来的图片送到关节点估计网络
        pose_hm = self.model(inps.to(self.device)).cpu().data
        # print("pose_hm:", pose_hm.shape)
        # Cut eyes and ears 以及下半身关节点
        # pose_hm = torch.cat([pose_hm[:, :1, ...], pose_hm[:, 5:13, ...]], dim=1)

        #  320 256
        xy_hm, xy_img, scores = getPrediction(pose_hm, pt1, pt2, self.inp_h, self.inp_w,
                                              pose_hm.shape[-2], pose_hm.shape[-1])
        # print('scores:', scores)
        result = pose_nms(bboxs, bboxs_scores, xy_img, scores)
        # print(result)
        return result
