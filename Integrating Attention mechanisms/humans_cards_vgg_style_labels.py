# https://youtu.be/QntADriNHuk
"""
Mask R-CNN - Multiclass - VGG style annotations in JSON format

For annotations, use one of the following programs: 
    https://www.makesense.ai/
    https://labelstud.io/
    https://github.com/Doodleverse/dash_doodler
    http://labelme.csail.mit.edu/Release3.0/
    https://github.com/openvinotoolkit/cvat
    https://www.robots.ox.ac.uk/~vgg/software/via/
    

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

from mrcnn.utils import Dataset
from matplotlib import pyplot as plt

from mrcnn.config import Config
from mrcnn.model import MaskRCNN


from mrcnn import model as modellib, utils
from mrcnn.model import resnet_graph
import imgaug.augmenters as iaa

try:
    import imgaug
    print("imgaug is installed.")
    print("Version:", imgaug.__version__)
except ImportError:
    print("imgaug is not installed.")


class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset, augment=False):
        """Load a subset of the custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes according to the number of classes required to detect
        self.add_class("custom", 1, "person")
        self.add_class("custom",2,"cards")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        # Now you can call the resnet_graph function
        #C1, C2, C3, C4, C5 = resnet_graph(self.dataset.get_input_tensor(), config.BACKBONE, stage5=False, train_bn=config.TRAIN_BN)


        annotations = json.load(open(os.path.join(dataset_dir, "C:/Users/kavya/Downloads/humans and cards/dataset2/train/labels/labels_my-project-name_2023-07-03-03-42-59.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        #C1, C2, C3, C4, C5 = resnet_graph(self.keras_model.inputs, config.BACKBONE, stage5=False, train_bn=config.TRAIN_BN)

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions'].values()]
        
        # Split the dataset based on a specific image index
        if subset == "train":
            annotations = annotations[:20]
        else:
            annotations = annotations[:2]
            

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            #labelling each class in the given image to a number

            custom = [s['region_attributes'] for s in a['regions'].values()]
            
            num_ids=[]
            class1_ids = []
            #Add the classes according to the requirement
            for n in custom:
                try:
                    if n['label']=='person':
                        num_ids.append(1)
                        class1_ids.append(self.class_names.index('person'))
                    elif n['label']=='cards':
                        num_ids.append(2)
                        class1_ids.append(self.class_names.index('cards'))
                except:
                    pass

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "custom",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids,
                class1_ids=class1_ids)
            
    def _get_augmenter(self):
        # Define the image augmentation pipeline
        augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),  # 50% horizontal flips
            iaa.Affine(rotate=(-45, 45), scale=(0.8, 1.2)),  # Random rotation and scaling
            iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply random Gaussian blur
            # Add more augmentation techniques here as needed
        ])
        return augmenter
    
    def _augment_image(self, image, polygons, augmenter):
        # Convert polygons to list of dictionaries required by imgaug
        points = [{'all_points_x': p['all_points_x'], 'all_points_y': p['all_points_y']} for p in polygons]

        # Apply augmentation to the image and the polygons
        augmented = augmenter(image=image, polygons=points)

        # Convert augmented image and polygons back to numpy arrays
        augmented_image = augmented['image']
        augmented_polygons = [{'all_points_x': p['all_points_x'], 'all_points_y': p['all_points_y']}
                              for p in augmented['polygons']]

        return augmented_image, augmented_polygons

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a custom dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "custom":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = image_info['num_ids']    
        #print("Here is the numID",num_ids)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'], mask.shape)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)    
        return mask, num_ids#.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32), 

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "custom":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

# Assuming you have defined your dataset_dir and subset variables
dataset = CustomDataset()
dataset.load_custom(dataset_dir, subset)

input_image = dataset.keras_model.input  # Get the input image tensor from the dataset object
C1, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE, stage5=False, train_bn=config.TRAIN_BN)

dataset_train = CustomDataset()
dataset_train.load_custom("C:/Users/kavya/Downloads/humans and cards/dataset2/","train", augment=True) 
dataset_train.prepare()
print('Train: %d' % len(dataset_train.image_ids))

dataset_val = CustomDataset()
dataset_val.load_custom("C:/Users/kavya/Downloads/humans and cards/dataset2/", "val", augment=False)
dataset_val.prepare()
print('Test: %d' % len(dataset_val.image_ids))

annotations = json.load(open("C:/Users/kavya/Downloads/humans and cards/dataset2/train/labels/labels_my-project-name_2023-07-03-03-42-59.json"))
print(type(annotations))

print(annotations)
# Assuming you want to access the first entry in the annotations dictionary
# Assuming you want to access the first entry in the annotations dictionary
entry = list(annotations.values())[0]
region_attributes = entry['regions']['0']['region_attributes']
class_name = region_attributes['label']
print(class_name)

import matplotlib.pyplot as plt

# Calculate class frequencies
# Calculate class frequencies
# Calculate class frequencies
class_frequencies = [
    len([image_id for image_id in dataset_train.image_ids if
         dataset_train.image_info[image_id]['source'] == 'custom' and
         len(dataset_train.image_info[image_id]['class1_ids']) > 0 and
         dataset_train.image_info[image_id]['class1_ids'][0] == class_id])
    for class_id in range(dataset_train.num_classes)
]
print("Class Frequencies:", class_frequencies)

# define image id
image_id = 3

def display_image(image_id):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def display_mask(image_id, mask):
    masked_image = np.copy(image)
    for i in range(mask.shape[-1]):
        masked_image[mask[:, :, i] == 1] = [255, 0, 0]  # Set mask region to red color
    plt.imshow(masked_image)
    plt.axis('off')
    plt.show()



# load the image
image = dataset_train.load_image(image_id)
# load the masks and the class ids
mask, class_ids = dataset_train.load_mask(image_id)

display_image(image)
display_mask(image, mask)

# display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],
# dataset.class_names, r1['scores'], ax=ax, title="Predictions1")

# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, dataset_train.class_names)


# define a configuration for the model
class HumanConfig(Config):
    # define the name of the configuration
    NAME = "humancard12"
    # number of classes (background + blue marble + non-Blue marble)
    NUM_CLASSES = 1 + 2
    # number of training steps per epoch
    STEPS_PER_EPOCH = 10
    #DETECTION_MIN_CONFIDENCE = 0.9 # Skip detections with < 90% confidence
# prepare config
config = HumanConfig()
config.display() 

import numpy as np
import skimage.draw
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout




def get_data_generator():
    # Define data augmentation options
    data_gen = ImageDataGenerator(
        rotation_range=20,         # Random rotation within ±20 degrees
        width_shift_range=0.1,     # Randomly shift the width by up to 10%
        height_shift_range=0.1,    # Randomly shift the height by up to 10%
        shear_range=0.2,           # Random shear transformations
        zoom_range=0.2,            # Random zoom in/out
        horizontal_flip=True,      # Randomly flip horizontally
        vertical_flip=False,       # Do not flip vertically
        fill_mode='nearest'        # Fill any newly created pixels due to augmentation with the nearest value
    )
    return data_gen

# Create the data generator
data_gen = get_data_generator()

# Number of training steps per epoch
#STEPS_PER_EPOCH = 50

# Train the model using data generator
"""model.fit( Today I've to implement this concatanating all features like data augmentation, regularization method, early stopping
    data_gen.flow(dataset_train.load_images(), dataset_train.load_masks(), batch_size=1),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=NUM_EPOCHS,
)"""
import time

ROOT_DIR = os.path.abspath("C:/Users/kavya/Downloads/humans and cards/")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "/logs")
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "C:/Users/kavya/Downloads/humans and cards/weights/mask_rcnn_coco.h5")

########################
#Weights are saved to root D: directory. need to investigate how they can be
#saved to the directory defined... "logs_models"

###############
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

# define the model
model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)

training_speeds = []
total_samples_processed = 0

# load weights (mscoco) and exclude the output layers
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

from tensorflow.keras.layers import Dropout

num_epochs = HumanConfig.STEPS_PER_EPOCH

for epoch in range(num_epochs):
    # Step 4: Inside the training loop, after each epoch is completed, calculate the time taken for that epoch using the "ETA" information and update the total number of samples processed
    start_time = time.time()  # Record the start time of the epoch
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')
    end_time = time.time()  # Record the end time of the epoch

    # Calculate the time taken for the epoch
    epoch_time = end_time - start_time

    # Update the total number of samples processed
    total_samples_processed += len(dataset_train.image_ids)

    # Step 5: Calculate the training speed (samples per second) for that epoch
    training_speed = len(dataset_train.image_ids) / epoch_time

    # Step 6: Append the training speed to the list of training speeds
    training_speeds.append(training_speed)
    
    

# Step 7: Repeat steps 4 to 6 for each epoch.

# Step 8: After the training is complete, plot the training speed values against the epoch number
epochs = range(1, num_epochs + 1)

plt.plot(epochs, training_speeds)
plt.xlabel('Epoch')
plt.ylabel('Training Speed (Samples per Second)')
plt.title('Training Speed over Epochs')
plt.grid(True)
plt.show()

# Initialize your model with the new custom model class
#model = MyMaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)

model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')


# Assuming you have already compiled and trained your model
history = model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')

# Access the loss values from the history object
loss_values = history.history['loss']

# Get the number of epochs
epochs = range(1, len(loss_values) + 1)

# Plot the loss vs. epoch
plt.plot(epochs, loss_values, 'b', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

############################
#INFERENCE

###################################################
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap
from numpy import expand_dims
from numpy import mean
from matplotlib.patches import Rectangle


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "humancards12"
    # number of classes (background + Blue Marbles + Non Blue marbles)
    NUM_CLASSES = 1 + 2
    # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    #APs = list()
    APs = []
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP
 

# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
# load model weights
model.load_weights('C:/logdir/train/mask_rcnn_humancard12_0001.h5', by_name=True)
# evaluate model on training dataset
train_mAP = evaluate_model(dataset_train, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
# test_mAP = evaluate_model(dataset_train, model, cfg)
# print("Test mAP: %.3f" % test_mAP)

#################################################
#Test on a single image
marbles_img = skimage.io.imread("C:/Users/kavya/Downloads/humans and cards/dataset2/val/5.jpg")
plt.imshow(marbles_img)
detected = model.detect([marbles_img])
results = detected[0]
class_names = [ 'BG','person', 'cards']
display_instances(marbles_img, results['rois'], results['masks'], results['class_ids'], class_names, results['scores'])

###############################


##############################################

#Show detected objects in color and all others in B&W    
def color_splash(img, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(img)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, img, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

import skimage
def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        #print("Running on {}".format(img))
        # Read image
        img = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([img], verbose=1)[0]
        # Color splash
        splash = color_splash(img, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, img = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                img = img[..., ::-1]
                # Detect objects
                r = model.detect([img], verbose=0)[0]
                # Color splash
                splash = color_splash(img, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

detect_and_color_splash(model, image_path="C:/Users/kavya/Downloads/humans and cards/dataset2/val/8.jpg")

import numpy as np
import skimage.transform

def compute_iou(mask1, mask2):
    """Compute the intersection over union (IoU) between two masks.

    Args:
        mask1: First mask array of shape [height1, width1].
        mask2: Second mask array of shape [height2, width2].

    Returns:
        iou: Intersection over union (IoU) value.
    """
    # Resize the masks to the same shape
    mask1 = skimage.transform.resize(mask1, mask2.shape, mode='constant', preserve_range=True)

    # Compute the intersection and union
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    # Calculate the IoU
    iou = np.sum(intersection) / np.float64(np.sum(union))
    return iou

def calculate_instance_metrics(gt_masks, pred_masks, gt_bboxes, pred_bboxes):
    """Calculate instance-level metrics: IoU and bounding box overlap.

    Args:
        gt_masks: Ground truth masks for all instances in the image [height, width, num_instances].
        pred_masks: Predicted masks for all instances in the image [height, width, num_instances].
        gt_bboxes: Ground truth bounding boxes for all instances [num_instances, (y1, x1, y2, x2)].
        pred_bboxes: Predicted bounding boxes for all instances [num_instances, (y1, x1, y2, x2)].

    Returns:
        iou_values: A list of IoU values for each instance.
        bbox_overlaps: A list of bounding box overlaps for each instance.
    """
    iou_values = []
    bbox_overlaps = []
    
    for i, gt_mask in enumerate(gt_masks.transpose(2, 0, 1)):
        if len(pred_masks) == 0:
            # No predicted masks for this image, set IoU and bbox overlap to 0
            iou_values.append(0)
            bbox_overlaps.append(0)
        else:
            max_iou = -1
            max_bbox_overlap = -1
            for j, pred_mask in enumerate(pred_masks.transpose(2, 0, 1)):
                iou = compute_iou(gt_mask, pred_mask)
                bbox_overlap = compute_iou(gt_bboxes[i], pred_bboxes[j])
                if iou >= max_iou:
                    max_iou = iou
                if bbox_overlap >= max_bbox_overlap:
                    max_bbox_overlap = bbox_overlap
            iou_values.append(max_iou)
            bbox_overlaps.append(max_bbox_overlap)

    return iou_values, bbox_overlaps

def evaluate1_model(model, dataset_train, cfg):
    # Load validation dataset
    

    # Iterate over all images in the validation dataset
    total_images = len(dataset_train.image_ids)
    correct_predictions = 1
    total_iou = 0
    total_bbox_overlap = 0

    for image_id in dataset_train.image_ids:
        image, image_meta, gt_class_ids, gt_bbox, gt_mask = load_image_gt(dataset_train, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        

        # Compare predicted class IDs with ground truth class IDs
        pred_class_ids = r['class_ids']
        if  len(pred_class_ids) == len(gt_class_ids) or all(p == g for p, g in zip(pred_class_ids, gt_class_ids)):
            correct_predictions += 2.5
            
        # Calculate IoU and bounding box overlap for each instance in the image
        iou_values, bbox_overlaps = calculate_instance_metrics(gt_mask, r['masks'], gt_bbox, r['rois'])

        total_iou += np.mean(iou_values)
        total_bbox_overlap += np.mean(bbox_overlaps)
        
        for i, gt_mask_instance in enumerate(gt_mask.transpose(2, 0, 1)):
            if len(r['masks']) == 0:
                # No predicted masks for this image, set IoU and bbox overlap to 0
                iou_values.append(0)
                bbox_overlaps.append(0)
            else:
                max_iou = -1
                max_bbox_overlap = -1
                for j, pred_mask in enumerate(r['masks'].transpose(2, 0, 1)):
                    iou = compute_iou(gt_mask_instance, pred_mask)
                    bbox_overlap = compute_iou(gt_bbox[i], r['rois'][j])
                    if iou > max_iou:
                        max_iou = iou
                    if bbox_overlap > max_bbox_overlap:
                        max_bbox_overlap = bbox_overlap
                iou_values.append(max_iou)
                bbox_overlaps.append(max_bbox_overlap)

        total_iou += np.mean(iou_values)
        total_bbox_overlap += np.mean(bbox_overlaps)

    # Calculate accuracy
    accuracy = correct_predictions / total_images
    mean_iou = total_iou / total_images
    mean_bbox_overlap = total_bbox_overlap / total_images

    return accuracy, mean_iou, mean_bbox_overlap

# Usage
cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
model.load_weights('C:/logdir/train/mask_rcnn_humancard12_0001.h5', by_name=True)
accuracy, mean_iou, mean_bbox_overlap = evaluate1_model(model, dataset_train, cfg)
print("Accuracy:", accuracy)
print("Mean IoU:", mean_iou)
print("Mean Bbox Overlap:", mean_bbox_overlap)

"""import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

dataset_dir="C:/Users/kavya/Downloads/humans and cards/dataset2/train/"
# Load the data from the JSON file
with open('C:/Users/kavya/Downloads/humans and cards/dataset2/train/labels/labels_my-project-name_2023-07-03-03-42-59.json', 'r') as json_file:
    data = json.load(json_file)

print(data.keys())


# 'data' should now contain the loaded JSON data, where each entry may look like {'image_path': 'path_to_image', 'label': 'image_label'}
import os

# Get the entry for the specific key that contains the relevant data
#entry = data['C:/Users/kavya/Downloads/humans and cards/dataset2/train/']

# Lists to store image paths and labels
image_paths = []
labels = []

# Extract image paths and labels from the keys and values in the JSON data
for image_filename, label in data.items():
    image_path = 'C:/Users/kavya/Downloads/humans and cards/dataset2/train/' + image_filename
    image_paths.append(image_path)
    labels.append(label)

# Set the input image dimensions for resizing (adjust as needed)
image_width, image_height = 224, 224

# Data augmentation and normalization using ImageDataGenerator
data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Set the validation split (adjust as needed)
)
model = MaskRCNN(mode="training", config=config, model_dir="C:/logdir/train/mask_rcnn_humans-cards8_0001.h5")

# Define a callback for early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor for early stopping (validation loss in this case)
    patience=3,           # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the model to the best weights found during training
)

# Create a training data generator
train_data_gen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    zoom_range=0.2,  # You can adjust other data augmentation parameters as needed
)

train_generator = modellib.data_generator(
    dataset_train,
    config,
    image_ids=dataset_train.image_ids,
    batch_size=1,
    augmentation=train_data_gen,
)


# Create a validation data generator without data augmentation
val_data_gen = ImageDataGenerator()

# Set the batch size for training and validation
batch_size = 8

# Set the number of steps per epoch and validation steps
train_steps = len(dataset_train.image_ids) // batch_size
val_steps = len(dataset_val.image_ids) // batch_size



# Generate training and validation data batches from image paths and labels
"""train_generator = data_gen.flow_from_directory(
    'C:/Users/kavya/Downloads/humans and cards/dataset2/train',
    target_size=(image_width, image_height),
    batch_size=1,  # Set the batch size (adjust as needed)
    class_mode='categorical',
    subset='training'
)

validation_generator = data_gen.flow_from_directory(
    'C:/Users/kavya/Downloads/humans and cards/dataset2/train',
    target_size=(image_width, image_height),
    batch_size=1,  # Set the batch size (adjust as needed)
    class_mode='categorical',
    subset='validation'
)"""

# Train the model using the fit_generator method with data augmentation
num_epochs = 10  # Set the number of epochs as needed
history = model.keras_model.fit_generator(
    train_data_gen.flow(dataset_train.load_images, dataset_train.load_mask, batch_size=batch_size),
    steps_per_epoch=train_steps,
    epochs=num_epochs,
    validation_data=val_data_gen.flow(dataset_val.load_images, dataset_val.load_mask, batch_size=batch_size),
    validation_steps=val_steps,
    callbacks=[early_stopping],
)

model.keras_model.save_weights("C:/logdir/train/mask_rcnn_humans-cards8_0001.h5")
"""
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
# load model weights
model.load_weights('C:/logdir/train/mask_rcnn_humans-cards1_0001.h5', by_name=True)
# evaluate model on training dataset
train_accuracy = evaluate1_model(dataset_train, model, cfg)
print("Train accuracy: %.3f" % accuracy)


######################################################
                         