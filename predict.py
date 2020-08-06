from typing import Union, List, Optional, Tuple
import json
import os
import glob
import cv2
import time
from colorama import Fore, Style
import argparse
import torch
import yaml
from tqdm import tqdm
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

USE_CUDA = True if torch.cuda.is_available() else False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

def restricted_float(x):
    """utility for command line options
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def load_model(weights_path: Union[str, os.PathLike],
               p_cfg_path: Union[str, os.PathLike],
               compound_coef: float) -> EfficientDetBackbone :
    """Loads and return model with given weights and project config.

    Args:
        weights_path (Union[str, os.PathLike]): Path to model weights.
        p_cfg_path (Union[str, os.PathLike]): Path to Project config yaml file.
        compound_coef (float): Compund scaling coefficient.

    Returns:
        EfficientDetBackbone: EfficientDet model
    """      
    params = yaml.safe_load(open(p_cfg_path))
    obj_list = params['obj_list']
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    if USE_CUDA:
        model.cuda()
    model.requires_grad_(False)
    model.eval()
    return model

def predict(images: List[Union[str, os.PathLike]],
        model: EfficientDetBackbone,
        compound_coef: float, 
        resize: Optional[Union[int, Tuple[int, int]]] = None,
        confidence: Optional[float] = 0.5,
        nms_threshold: Optional[float] = 0.5,
        output_path: Union[str, os.PathLike] = "../",
    ) -> None:
    """Generate Predictions on test images in a folder.

    Args:
        images (List[Union[str, os.PathLike]]): List of test image path to run predictions.
        model (EfficientDetBackbone): EfficientDet model.
        compound_coef (float): Compund scaling coefficient.
        resize (Optional[Union[int, Tuple[int, int]]], optional): Resize of test images. Defaults to None.
        confidence (Optional[float], optional): confidence score to filter detections. Defaults to 0.5.
        nms_threshold (Optional[float], optional): IOU threshold to filter duplicate detections. Defaults to 0.5.
        output_path (Union[str, os.PathLike], optional): Output path/file where final output needs to be stored. Defaults to "../".

    Raises:
        IOError: Raises when output_path do not exist.
    """      

    #Initializaing results
    results = {}
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    
    #Iterating over all images
    for image_path in tqdm(images):
        #Initalize and Get image name.
        img_result = []
        img_name = image_path.split('/')[-1]

        #Preprocess image
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=INPUT_SIZES[compound_coef])
        x = torch.from_numpy(framed_imgs[0])
        
        #Convert to CUDA or CPU.
        if USE_CUDA:
            x = x.cuda()
            x = x.float()
        else:
            x = x.float()

        #Batching
        x = x.unsqueeze(0).permute(0, 3, 1, 2)

        #Run model
        features, regression, classification, anchors = model(x)

        #Applying threshold and NMS on predictions
        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            confidence, nms_threshold)
        
        #Continue if there are no predictions for this image.
        if not preds:
            results[img_name] = img_result
            continue
        
        #Convert predictions.
        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        #Convert bbox and others to required format.
        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            #rois[:, 2] -= rois[:, 0]
            #rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                img_result.append({
                    'class_index': label,
                    'bbox': box.tolist(),
                    'confidence': float(score)
                })

                results[img_name] = img_result

    if not len(results):
        print('The model does not provide any valid output, check model architecture and the data input')

    # Write output
    if output_path.endswith(".json"):
        if os.path.exists(os.path.dirname(output_path)):
            output_file = output_path
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_file = output_path
    elif os.path.isdir(output_path):
        output_file = os.path.join(
            output_path, "yolov5_predictions_" + str(time.time()).split(".")[0] + ".json"
        )

    else:
        raise IOError(
            f"{Fore.RED} no such directory {os.path.dirname(output_path)} {Style.RESET_ALL}"
        )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detections are written to {output_file}.")


if __name__ == "__main__":
    """
        Driver code
    """
    #Parse Arguments    
    parser = argparse.ArgumentParser(description="EfficientDet prediction")
    parser.add_argument("-i", "--image-dir", help="path to the directory with test images")
    parser.add_argument("-w", "--weights", help="path to the weights file")
    parser.add_argument("-o", "--output-path", help="path to output file/directory")
    parser.add_argument("-ccf", "--compound-coef", help="Compound coeff",default=1,type=int,required=True)
    parser.add_argument("-pcp", "--project-config-path", help="Project config path")
    parser.add_argument(
        "--confidence",
        type=restricted_float,
        required=False,
        default=0.5,
        help="minimum confidence score for the predictions",
    )
    parser.add_argument(
        "--nms-threshold",
        type=restricted_float,
        required=False,
        default=0.5,
        help="IOU threshold for non-max-suppression",
    )
    parser.add_argument(
        "--resize",
        required=False,
        help="resize the images to a specified dimension before predictions",
    )
    parser.add_argument(
        "--file-ext", required=False, default="jpg", help="file extension of the images"
    )

    opt = parser.parse_args()

    #Parsing resize argument
    resize = opt.resize
    if resize is not None:
        resize = tuple(map(int, resize.split(",")[:2]))
        if len(resize) == 1:
            resize = resize[0]

    #Load model
    model = load_model(opt.weights,opt.project_config_path,opt.compound_coef)

    #Get image paths
    images = glob.glob(os.path.join(opt.image_dir, "*." + opt.file_ext))

    #Run Predictions        
    predict(
        images,
        model,
        opt.compound_coef,
        resize,
        opt.confidence,
        opt.nms_threshold,
        opt.output_path
    )
