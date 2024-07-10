import os
import cv2
import time
import torch
import argparse
import numpy as np
import shutil
from fn import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
# from DetectorLoader_V3_tiny import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single
from Detection.yolox.yolo import YOLO
from ActionsEstLoader import TSSTG
import warnings
from Track.Tracker import Detection, Tracker
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
video_name = '001.mp4'
pre_path = r'C:/Users/jj/Documents/咸鱼业务/12.4/action_kalman/Data/video'
source = os.path.join(pre_path,video_name)
output_name = video_name.split('.')[0] + '.txt'
abnormal_log = os.path.join('./data/output', output_name)
# 调用相机
# source = "1"


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Workflow recognition')
    par.add_argument('--source', default='Data/video/002.mp4',  # required=True,  # default=2,
                     help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=640,
                     help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_model', type=str, default='hrnet',
                     help='pese model, include hrnet & alpha-pose & hr_trt')
    par.add_argument('--show_detected', default=False, action='store_true',
                     help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                     help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='001.avi',
                     help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                     help='Device to run model on cpu or cuda.')
    par.add_argument('--detect_model', type=str, default='yolox', help="model for detection: yolox or V3_tiny")
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    print("using {} to detect".format(args.detect_model))
    # if args.detect_model == 'V3_tiny':
    #     detect_model = TinyYOLOv3_onecls(inp_dets, nms=0.6, conf_thres=0.3, device=device)
    # else:
    detect_model = YOLO(device=device)



    # POSE MODEL
    # 224*160 alphapose  384*288 for hrnet (must be divided by 32)
    inp_pose = '224x160' if args.pose_model == 'alpha_pose' else '384x288'
    inp_pose = inp_pose.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))

    pose_model = SPPE_FastPose(args.pose_model, inp_pose[0], inp_pose[1], device=device)
    tracker = Tracker(max_age=30, n_init=3)

    #  Actions Estimate.

    action_model = TSSTG(device=device)

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.source
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file. type: BGR
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    # ---------------------------------------------创建保存日志的文件夹----------------------------------------------------
    if os.path.exists("./log.txt"):
        os.remove("./log.txt")

    if os.path.exists("./logs"):
        shutil.rmtree("./logs")
    os.mkdir("./logs")

    if os.path.exists(abnormal_log):
        os.remove(abnormal_log)
    # if os.path.exists(abnormal_log):
    #     os.remove(abnormal_log)
    # os.mkdir(abnormal_log)

    if type(cam_source) is str and os.path.isfile(cam_source):
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess= None if args.detect_model == 'yolox' \
        else preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source, \
                        None if args.detect_model == 'yolox' else preproc).start()
    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (3840, 2160))
    frame_ = 0
    while cam.grabbed():
        frame_ += 1  # 视频的帧数
        video_fps, video_frame_count = cam.fps_count()
        frame = cam.getitem()
        image = frame.copy()
        # Detect humans bbox in the frame with detector model.
        t_begin = time.time()
        detected = detect_model.detect_image(frame)
        tracker.predict()
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [1.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        t_det = time.time()
        # print("det_time:", t_det - t_begin)
        detections = []
        if detected is not None:
            poses = pose_model.predict(image, detected[:, 0:4], detected[:, 4])
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 1)
        tracker.update(detections)
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)
            clr = (0, 255, 0)
            action = 'Uninit'
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, image.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                if action_name == "touch" or action_name == "climb":
                    clr = (255, 0, 0)
                    with open("./logs/person_{}.txt".format(track_id), "a") as f:
                        f.write("ActionAbnormal:{},PersonID:{},In{}s\n".format(action_name, track_id, frame_/video_fps))
                    f.close()
                # print("----->>>>>>", action_name)
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    if len(track.keypoints_list):
                        frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (frame_, 1.0 / (time.time() - t_begin)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]

        if outvid:
            # print(frame.shape)
            writer.write(frame)
# ----------------------窗口自适应------------------------------
        cv2.namedWindow('picture', 0)
        cv2.imshow("picture", frame)
# ----------------------不加窗口自适应------------------------------
#         cv2.namedWindow("frame", 0)
        # cv2.imshow('frame', frame)
#         cv2.waitKey(3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
    log_list = os.listdir("./logs")
    for log in log_list:
        time_1 = [-1]
        with open("./logs/{}".format(log), "r") as f:
            lines = f.read().splitlines()
            for i in range(len(lines) - 1):
                if float(lines[i + 1].split("In")[1].split("s")[0]) - float(
                        lines[i].split("In")[1].split("s")[0]) > 0.04 or \
                        lines[i + 1].split(":")[1].split(",")[0] != lines[i].split(":")[1].split(",")[0]:
                    time_1.append(i)
            time_1.append(len(lines) - 1)
        for i in range(len(time_1) - 1):
            with open(abnormal_log, 'a') as f1:
                f1.write("人员ID:{},开始时间:{},结束时间:{},违规动作:{}\n".format(log.split(".")[0], \
                             lines[time_1[i]+1].split("In")[1], lines[time_1[i + 1]].split("In")[1], \
                                                                    lines[time_1[i] + 1].split(":")[1].split(",")[0]))
