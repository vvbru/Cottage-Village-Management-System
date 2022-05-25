# -*- coding: utf-8 -*-
import logging
import argparse
import cv2
import numpy as np
import tensorflow as tf
from pytesseract import image_to_string
from path import Path
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from paho.mqtt import client as mqtt_client
from threading import Thread

broker = 'dev.rightech.io'
port = 1883
client_id = 'mqtt-vvburdyug-e3wiy9'

delay = 0

logging.getLogger('tensorflow').setLevel(logging.ERROR)

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
#    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def open(client):
    client.publish('base/state/gate', True)
    global delay
    delay = 20
    while delay > 0:
        print(delay)
        delay = delay - 1
        time.sleep(1)

    client.publish('base/state/gate', False)


def detection_params(image, detection_graph, sess):
    """
    Extract detection params
    Parameters
    ----------
    image: cv image
        Image with detected numbers
    detection_graph: tf graph
    sess: tf session

    Return
    ---------------
        params of detection such as boxes, scores, classes, num_detections - lists and tuples
    """
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image, axis=0)
    # Extract image tensor
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Extract detection boxes
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Extract detection scores
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # Extract detection classes
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Extract number of detections
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                        feed_dict={image_tensor: image_np_expanded})

    return boxes, scores, classes, num_detections


def ocr_tesseract(image):
    """
    OCR function with tesseract on board
    Parameters
    --------------
    image: cv image
        image with car number plate on it

    Return
    ---------------
    numbers: str with car number
    """
    img_after_resize = cv2.resize(image, None, fx=2, fy=2,
                                  interpolation=cv2.INTER_CUBIC)

    img_after_bilateral_filter = cv2.bilateralFilter(img_after_resize, 10, 20, 20)
    img_gray = cv2.cvtColor(img_after_bilateral_filter, cv2.COLOR_BGR2GRAY)
    # ret3, th3 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    numbers = image_to_string(img_gray, lang='rus', config='--psm 6'
                                                           '--oem 0'
                                                           '-c tessedit_char_whitelist=0123456789ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁ'
                                                           '-c tessedit_char_blacklist=.,|><:;~`/][}{-=)($%^-!')
    return numbers


def cut_detect_numbers(scores, boxes, image_np):
    """
    Cut numbers from image and OCR them
    Parameters
    -------------
    scores: scores of detection
    boxes: detection bounding boxes
    image_np: cv image with detected car-plate number

    Return
    -------------
    numbers: list with detected numbers as str
    """
    numbers = []

    for i in range(scores[0].shape[0]):
        if int(100 * scores[0][i]) > 50:

            img_screenshot = image_np
            # Get coordinates of detection bbox
            box = boxes[0][i]
            dx = int(box[1] * img_screenshot.shape[1])
            dy = int(box[0] * img_screenshot.shape[0])
            dw = int((box[3] - box[1]) * img_screenshot.shape[1])
            dh = int((box[2] - box[0]) * img_screenshot.shape[0])

            crop_img = img_screenshot[dy:dy + dh, dx:dx + dw]
            number = ocr_tesseract(crop_img)
            numbers.append(number)
    return numbers


def detector(detection_graph, category_index):
    """
    Make car number detection

    Parameters
    ------------------------
    detection_graph: tf graph
    category_index: Index of categories
    mode: Image or video mode
    image_path: Path to the image
    ------------------------
    """

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Video mode
            cap = cv2.VideoCapture('123.m4v')
            while True:
                # Read frame from camera
                ret, image_np = cap.read()
                boxes, scores, classes, num_detections = detection_params(image_np, detection_graph, sess)

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np, np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2)

                numbers = cut_detect_numbers(scores, boxes, image_np)
                #print('Detected numbers --- \n', numbers, '\n')

                d1 = datetime.date.today()
                df = pd.read_csv("data.csv")

                if numbers in df['Номер'].values:
                    if datetime.date(df.loc[df['Номер'] == numbers,'Год'],
                    df.loc[df['Номер'] == numbers,'Месяц'],
                    df.loc[df['Номер'] == numbers,'День']) >= d1:
                        th = Thread(target=open, args=(client, ))
                        global delay
                        if delay == 0:
                            th.start()
                        else:
                            delay = 20


def main():

    client = connect_mqtt()

    time.sleep(1)

    client.publish('base/state/gate', False)

    client.loop_start()

    # ----------------------------import model-------------------------------------------------------
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT =  Path('detector_model/frozen_inference_graph.pb')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS =  Path('object_detection.pbtxt')

    # Number of classes to detect
    NUM_CLASSES = 1

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # -----------------------------------------------------------------------------------------------
    # Apply detector
    detector(detection_graph, category_index)


if __name__ == '__main__':
    main()
