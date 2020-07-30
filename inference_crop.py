import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

from detectron2.data.datasets import register_coco_instances
#register_coco_instances("circle_dataset_train", {}, "coco/annotations/instances_train2017.json", "coco/images/train2017")
register_coco_instances("circle_dataset_val", {}, "coco/annotations/instances_val2017.json", "coco/images/val2017")


#MetadataCatalog.get("circle_dataset_train").set(thing_classes=["circle"])

# balloon_metadata = MetadataCatalog.get("circle_dataset_train")

#dataset_dicts = DatasetCatalog.get("circle_dataset_train")



from detectron2.engine import DefaultTrainer

cfg = get_cfg()

#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("circle_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 6000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

cfg.OUTPUT_DIR = 'wodeopt'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
cfg.DATASETS.TEST = ("circle_dataset_val", )
predictor = DefaultPredictor(cfg)


from detectron2.utils.visualizer import ColorMode
#dataset_dicts = get_balloon_dicts("balloon/val")
balloon_metadata = MetadataCatalog.get("circle_dataset_val")
dataset_dicts = DatasetCatalog.get("circle_dataset_val")
for d in random.sample(dataset_dicts, 12):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(outputs)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes
    boxes = np.int16(boxes.tensor.numpy())
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    filename = os.path.basename(d["file_name"])
    print(os.path.join('opt', filename))
    #cv2.imwrite(os.path.join('opt', filename), out.get_image()[:, :, ::-1])
    crop_image = im[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]]
    r = min(crop_image.shape[:-1]) // 2
    # crop inner circle
    print(crop_image.shape, crop_image.shape[:-1])
    mask = np.zeros(crop_image.shape[:-1], dtype=np.uint8)
    cv2.circle(mask, (crop_image.shape[0]//2, crop_image.shape[1]//2), r-10, 255, thickness=-1)
    circle_img = cv2.bitwise_and(crop_image, crop_image, mask=mask)

    cv2.imwrite(os.path.join('crop', filename), circle_img)
    # break

    # cv2.namedWindow('val', cv2.WINDOW_NORMAL)
    # cv2.imshow('val', out.get_image()[:, :, ::-1][boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]])
    # if cv2.waitKey(0) == 27:
    #     break


# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# evaluator = COCOEvaluator("balloon_val", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "balloon_val")
# print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way is to use trainer.test
