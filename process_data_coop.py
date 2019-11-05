import os
import tensorflow as tf
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import glob
import shutil
from PIL import Image

class Process_data:
    def __init__(self, path_pb='/media/HDD/Download/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_pb, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True

            self.sess = tf.Session(graph=detection_graph, config=config)
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0') # place holder

    def detect(self, path_img):
        rawimg = np.array(Image.open(path_img).convert('RGB'))
        r_pad = 0.5
        h_raw, w_raw, _ = rawimg.shape
        maxsize_raw = int(max(h_raw,w_raw)*(1+r_pad))
        pad_left = int((maxsize_raw-w_raw)//2)
        pad_right = maxsize_raw-w_raw-pad_left
        pad_top = int((maxsize_raw-h_raw)//2)
        pad_bot = maxsize_raw-h_raw-pad_top
        img = np.pad(rawimg, ((pad_top, pad_bot), (pad_left, pad_right), (0,0)), constant_values=255)

        output_dict = self.sess.run(self.tensor_dict,
            feed_dict={self.image_tensor: np.expand_dims(img, 0)})

        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        use_classid = [4] # person car bike  truck
        filter_id = [idx for idx in range(len(output_dict['detection_classes'])) if ((output_dict['detection_classes'][idx] in use_classid) and (output_dict['detection_scores'][idx]>0.9 ))]
        output_dict['detection_boxes'] = output_dict['detection_boxes'][filter_id]
        boxes = np.array(output_dict['detection_boxes'])
        im_height, im_width, _ =  img.shape
        boxes = (boxes*[im_height,im_width,im_height, im_width]).astype(np.int)
        #debug
        for y1,x1,y2,x2 in boxes:
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 5)
        #debug
        boxes = boxes - np.array([pad_top, pad_left, pad_top, pad_left])
        return boxes.astype(np.int), img
        





