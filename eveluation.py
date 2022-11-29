from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


cocoAnnotation = COCO(annotation_file="coco_eval/GT_result.json")
cocovalPrediction = cocoAnnotation.loadRes("coco_eval/Pred_result.json")

cocoEval = COCOeval(cocoAnnotation, cocovalPrediction, "bbox")

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()