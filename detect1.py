import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import time
import torch
import cv2
import pygame
import mediapipe as mp

# Initialize pygame audio
pygame.mixer.init()

# Object coordinates
left_obj_coords = (251, 202, 294, 351)
right_obj_coords = (525, 192, 561, 342)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

# Pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def is_facing_object(keypoints, obj_coords):
    """Determine if a person is facing the object based on keypoints."""
    nose = keypoints.get('nose')
    left_eye = keypoints.get('left_eye')
    right_eye = keypoints.get('right_eye')
    
    if not nose or not left_eye or not right_eye:
        return False

    # Calculate the center of the object
    obj_center_x = (obj_coords[0] + obj_coords[2]) / 2
    obj_center_y = (obj_coords[1] + obj_coords[3]) / 2
    
    # Calculate direction vector from nose to eyes
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    face_direction_x = eye_center_x - nose[0]
    face_direction_y = eye_center_y - nose[1]
    
    # Calculate direction vector from nose to object
    obj_direction_x = obj_center_x - nose[0]
    obj_direction_y = obj_center_y - nose[1]
    
    # Calculate dot product to determine if facing the object
    dot_product = face_direction_x * obj_direction_x + face_direction_y * obj_direction_y
    return dot_product > 0

def get_pose_keypoints(image):
    """Get pose keypoints from the image using MediaPipe."""
    keypoints = {}
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = {
            'nose': (landmarks[0].x, landmarks[0].y),
            'left_eye': (landmarks[1].x, landmarks[1].y),
            'right_eye': (landmarks[2].x, landmarks[2].y),
        }
    return keypoints

is_playing = False

def control_audio_playback(is_within_threshold, alert_sound):
    global last_within_threshold_time, last_outside_threshold_time, is_playing

    current_time = time.time()

    if is_within_threshold:
        if not is_playing and current_time - last_within_threshold_time >= 2:
            pygame.mixer.music.load(alert_sound)
            pygame.mixer.music.play()
            is_playing = True
            print('play')  # Resume audio playback
        last_within_threshold_time = current_time
        last_outside_threshold_time = 0
    else:
        last_outside_threshold_time = current_time

    # Pause playback if outside threshold for more than 2 seconds
    if is_playing and current_time - last_within_threshold_time > 2:
        pygame.mixer.music.pause()
        is_playing = False
        print('pause')

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Warmup
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    last_within_threshold_time = 0
    last_outside_threshold_time = 0

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
            )

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = (
                path[i],
                im0s[i].copy(),
                dataset.count
                if webcam
                else (0, getattr(dataset, "frame", 0)),
            )

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                            -1
                        ).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_crop:
                        save_one_box(
                            xyxy, im0, file=save_dir / "crops" / names[int(cls)] / f"{p.stem}.jpg", BGR=True
                        )

                    c = int(cls)  # integer class
                    label = None if hide_labels else (
                        names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                    )
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # Custom code to check proximity and play audio
                    if names[c] == "person":
                        x_min, y_min, x_max, y_max = xyxy
                        center_x = (x_min + x_max) / 2
                        center_y = (y_min + y_max) / 2
                        threshold_distance = 20

                        # Check if the person's coordinates are near either of the specified coordinates
                        near_object = is_near_object(center_x, center_y, left_obj_coords, right_obj_coords, threshold_distance)

                        # Get pose keypoints
                        keypoints = get_pose_keypoints(im0)

                        if near_object == "left" and is_facing_object(keypoints, left_obj_coords):
                            control_audio_playback(True, "sound1.wav")
                        elif near_object == "right" and is_facing_object(keypoints, right_obj_coords):
                            control_audio_playback(True, "sound2.wav")
                        else:
                            control_audio_playback(False, "")

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),  # H264
                            fps,
                            (w, h),
                        )
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        LOGGER.info("Results saved to %s" % save_dir)
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    check_requirements(exclude=("tensorboard", "thop"))
    run(**vars(opt))
