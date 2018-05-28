"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That"s a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, 
                  return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/person_keypoints_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)


        # Background is always the first class
        self.kp_class_info = [{"source": "", "id": 0, "name": "MISSING"}]
        self.kp_source_class_ids = {}

        # All person images
        image_ids = list(coco.getImgIds(catIds=[1]))

        # Add person class
        self.add_class("coco", 1, "person")

        # Add keypoint classes
        keypoint_names = coco.loadCats(1)[0]["keypoints"]
        for i, kpname in enumerate(keypoint_names):
            self.add_kp_class("coco", i+1, kpname)

        # Store skeleton (for visualization)
        self.skeleton = coco.loadCats(1)[0]["skeleton"]

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]["file_name"]),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], iscrowd=None)))
                    
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn"t exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, "wb") as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/person_keypoints_minival2014.json.zip".format(dataDir)
            annFile = "{}/person_keypoints_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/person_keypoints_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/person_keypoints_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/person_keypoints_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/person_keypoints_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/person_keypoints_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, "wb") as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id, scale=1.0, padding=[(0, 0), (0, 0), (0, 0)], crop=None, verify_size=True):
        """Loads the binary masks for each keypoint in the image.

        Returns:
        kp_masks: A bool array of shape [person_count, height, width, kp_count] with
            a one-mask per keypoint.
        kp_ids: an array of shape [person_count, kp_count] of keypoint IDs
            of the each person's keypoints.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)
        
        kp_masks = []
        kp_ids = []
        annotations = self.image_info[image_id]["annotations"]

        if crop: raise Exception("Crop not supported")

        # Calculate mask size
        pad_top, pad_bot, pad_left, pad_right = padding[0][0], padding[0][1], padding[1][0], padding[1][1]
        mask_size = (int(round(image_info["height"] * scale) + pad_top + pad_bot),
                     int(round(image_info["width"] * scale) + pad_left + pad_right))

        if verify_size and (mask_size[0] != 1024 or mask_size[1] != 1024):
            print("padding:",padding)
            print("scale:",scale)
            print("mask_size:",mask_size)
            raise Exception("Wrong maks size")

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            # Get keypoint positions
            kps = np.array(annotation["keypoints"])
            x = kps[0::3]
            y = kps[1::3]
            v = kps[2::3]
            masks = []
            ids = []
            for kpidx in range(1, self.num_kp_classes+1):
                mask = np.zeros(mask_size, dtype=bool)
                kp_id = 0
                
                # Consider only annotated keypoints
                if v[kpidx - 1] > 0:
                    kp_id = self.kp_map_source_class_id("coco.{}".format(kpidx))
                    if kp_id:
                        mask[int(round(y[kpidx - 1] * scale) + pad_top), int(round(x[kpidx - 1] * scale) + pad_left)] = True
                masks.append(mask)
                ids.append(kp_id)
            kp_masks.append(masks)
            kp_ids.append(ids)

        if kp_ids:
            # Append masks and kp ids
            kp_masks =  np.transpose(np.array(kp_masks).astype(np.bool), [0, 2, 3, 1])
            kp_ids = np.array(kp_ids)
        else:
            # Append an empty mask
            kp_masks = np.empty([0, 0, 0, 0])
            kp_ids = np.empty([0], np.int32)
        return np.array(kp_masks), np.array(kp_ids)

    def load_bbox(self, image_id, scale=1.0, padding=[(0, 0), (0, 0), (0, 0)], crop=None):
        """Load instance bounding box for the given image.

        Returns:
        bboxes: An array of shape [num_instance, (y1, x1, y2, x2)]
            with one bounding box per instance
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_bboxes = []
        annotations = self.image_info[image_id]["annotations"]

        # Get padding
        pad_top, pad_left = padding[0][0], padding[1][0]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            bbox = np.array([pad_top + y * scale, pad_left + x *scale, pad_top + (y + h) * scale, pad_left + (x + w) * scale])
            instance_bboxes.append(bbox)

        # Pack instance masks into an array
        if instance_bboxes:
            return np.stack(instance_bboxes, axis=0).astype(np.int32)
        else:
            # Call super class to return an empty mask
            return np.empty([0, 0])#super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

                # The following two functions are from pycocotools with a few changes.
    
    def add_kp_class(self, source, kp_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"

        # Does the class exist already?
        for info in self.kp_class_info:
            if info['source'] == source and info["id"] == kp_id:
                # source.class_id combination already available, skip
                return

        # Add the class
        self.kp_class_info.append({
            "source": source,
            "id": kp_id,
            "name": class_name,
        })

    def kp_map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.kp_map_source_class_id("coco.12") -> 23
        """
        return self.kp_class_from_source_map[source_class_id]

    def prepare(self, class_map=None):
        super(CocoDataset, self).prepare(class_map)
        
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_kp_classes = len(self.kp_class_info) - 1
        self.kp_class_ids = np.arange(self.num_kp_classes + 1)
        self.kp_class_names = [clean_name(c["name"]) for c in self.kp_class_info]

        # Mapping from source class and image IDs to internal IDs
        self.kp_class_from_source_map = { "{}.{}".format(info['source'], info['id']): id
                                          for info, id in zip(self.kp_class_info, self.kp_class_ids) }

        # Map sources to class_ids they support
        self.kp_sources = list(set([i['source'] for i in self.kp_class_info]))
        self.kp_source_class_ids = {}

        # Loop over datasets
        for source in self.kp_sources:
            self.kp_source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.kp_class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.kp_source_class_ids[source].append(i)

############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it"s the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN on MS COCO.")
    parser.add_argument("command",
                        metavar="<command>",
                        help="\"train\" or \"evaluate\" on MS COCO")
    parser.add_argument("--dataset", required=True,
                        metavar="/path/to/coco/",
                        help="Directory of the MS-COCO dataset")
    parser.add_argument("--year", required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help="Year of the MS-COCO dataset (2014 or 2017) (default=2014)")
    parser.add_argument("--model", required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or \"coco\"")
    parser.add_argument("--logs", required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help="Logs and checkpoints directory (default=logs/)")
    parser.add_argument("--limit", required=False,
                        default=500,
                        metavar="<image count>",
                        help="Images to use for evaluation (default=500)")
    parser.add_argument("--download", required=False,
                        default=False,
                        metavar="<True|False>",
                        help="Automatically download and unzip MS-COCO files (default=False)",
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we"ll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
        dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(args.dataset, "minival", year=args.year, auto_download=args.download)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers="heads",
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers="4+",
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers="all",
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        coco = dataset_val.load_coco(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("\"{}\" is not recognized. "
              "Use \"train\" or \"evaluate\"".format(args.command))
