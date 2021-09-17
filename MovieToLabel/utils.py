# import matplotlib
# import matplotlib.pyplot as plt

import io
import gc
import os
import cv2
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  del img_data
  gc.collect()
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_keypoint_tuples(eval_config):
  """Return a tuple list of keypoint edges from the eval config.
  
  Args:
    eval_config: an eval config containing the keypoint edges
  
  Returns:
    a list of edge tuples, each in the format (start, end)
  """
  tuple_list = []
  kp_list = eval_config.keypoint_edge
  for edge in kp_list:
    tuple_list.append((edge.start, edge.end))
  return tuple_list

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    del image
    gc.collect()

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn


def label_out(classes, scores, label_map_dict ,threshold=0.5):
  output = []
  for i in range(len(classes)):
    if scores[i] > threshold:
      keys = [k for k, v in label_map_dict.items() if v == classes[i]]
      # print(keys)
      output+=keys
    else:
      break
  del scores, classes, label_map_dict, threshold
  gc.collect()
  return output


def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            if n%20 == 0:
                tn = n//20
                cv2.imwrite('{}_{}.{}'.format(base_path, str(tn).zfill(digit), ext), frame)
            n += 1
        else:
            del ret, frame, tn, n
            gc.collect()
            return


def movie_to_label(image_path, threshold=0.5):
  #構成情報ファイルのパス。リポジトリにConfigファイルの揃ったフォルダがあるけど、微妙にモデル名が省略されていたりするので、ダウンロードしたものの方が確実？
  pipeline_config = "./centernet_hg104_512x512_coco17_tpu-8/pipeline.config"
  #チェックポイントのパス
  model_dir = "./centernet_hg104_512x512_coco17_tpu-8/checkpoint"

  #モデル構成情報読み込み
  configs = config_util.get_configs_from_pipeline_file(pipeline_config)
  model_config = configs['model']

  #読み込んだ構成情報でモデルをビルド
  detection_model = model_builder.build(
        model_config=model_config, is_training=False)

  #チェックポイントから重みを復元
  ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
  ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()
  detect_fn = get_model_detection_function(detection_model)

  label_map_path = './MovieToLabel/object_detection/data/mscoco_label_map.pbtxt'
  label_map = label_map_util.load_labelmap(label_map_path)
  categories = label_map_util.convert_label_map_to_categories(
      label_map,
      max_num_classes=label_map_util.get_max_label_map_index(label_map),
      use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

  # image_dir = '/content/drive/MyDrive/pjSOMA/バナナの画像/'
  # image_path = os.path.join(image_dir, 'バナナの画像_000.jpg')
  image_np = load_image_into_numpy_array(image_path)

  # Things to try:
  # Flip horizontally
  # image_np = np.fliplr(image_np).copy()

  # Convert image to grayscale
  # image_np = np.tile(
  #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

  input_tensor = tf.convert_to_tensor(
      np.expand_dims(image_np, 0), dtype=tf.float32)
  detections, predictions_dict, shapes = detect_fn(input_tensor)

  label_id_offset = 1
  image_np_with_detections = image_np.copy()

  # Use keypoints if available in detections
  keypoints, keypoint_scores = None, None
  if 'detection_keypoints' in detections:
    keypoints = detections['detection_keypoints'][0].numpy()
    keypoint_scores = detections['detection_keypoint_scores'][0].numpy()

  #ラベルを出力
  output = label_out((detections['detection_classes'][0].numpy() + label_id_offset).astype(int),detections['detection_scores'][0].numpy(), label_map_dict, threshold)
  del detections, label_id_offset, label_map_dict, threshold, input_tensor, predictions_dict, shapes, image_np_with_detections, keypoints, keypoint_scores, pipeline_config, model_dir, image_np, ckpt, detect_fn
  gc.collect()
  return output

  '''
  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=get_keypoint_tuples(configs['eval_config']))

  print(box_to_display_str_map)

  plt.figure(figsize=(96,128))
  plt.imshow(image_np_with_detections)
  plt.show()
  '''


def movie_to_text(movie_path, img_dir="temp", img_name="temp", threthold=0.5):
  print("Start save frames...")
  save_all_frames(movie_path, img_dir, img_name)
  frames = glob.glob(img_dir + "/*.jpg")
  output = []
  for i in range(len(frames)):
    print(f"[{str(i)}/{str(len(frames))}] Start...")
    output += movie_to_label(frames[i])
  del frames, movie_path, img_dir, img_name, threthold
  gc.collect()
  return list(set(output))