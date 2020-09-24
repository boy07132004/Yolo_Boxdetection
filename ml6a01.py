import argparse
import os
import darknet
import time
import cv2
import darknet
from Lora import lora
from box_count import box_count

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))


def image_detection(network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = 1280#darknet.network_width(network)
    height = 720#darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    for _ in range(5):cap.read()
    image = cap.read()[1]
    # Uncomment next line for local image detection
    image = cv2.imread("test.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cap.release()
    
    darknet.copy_image_from_bytes(darknet_image, image.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    # Draw boxes
    image = darknet.draw_boxes(detections, image_rgb, class_colors)
    cv2.imwrite("predictions.jpg",cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
    return detections

def main():
    args = parser()
    check_arguments_errors(args)

    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights
    )
    _lora = lora()

    while True:
        detections = image_detection(network, class_names, class_colors, args.thresh)
        # Do something before next detection
        box1,box2 = box_count(detections)
        _lora.send(f"Area1: {box1}, Area2: {box2}")
        print(box1,box2)
        print("Press 'Enter' to Continue")
        a= input()


if __name__ == "__main__":
    main()

