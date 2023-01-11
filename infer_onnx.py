import argparse
import cv2
import numpy as np
import logging as log
import sys, json
import utils
import time
from pathlib import Path

import onnxruntime as rt

log.basicConfig(
    format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout
)

class OnnxInfer:
    def __init__(self, onnx_file, model_config=None, device="CPU"):
        """

        Args:
            model_path (str): model .xml file
            input_size (int): with = height
            model_config (str): model config .json
        """
        config_path = Path(onnx_file).with_suffix(".json")

        self.onnx_file = onnx_file
        self.device = device.lower()

        self.config = self.load_config(config_path=config_path)
        self.input_size = self.__get_input_size(self.config)  # [h, w, c]
        assert len(self.input_size) == 3, "length of input size must be 3"

        self.model = self.__init_model()
        self.class_names = self.config["model_config"]["class_name"]
        self.input_name = self.model.get_inputs()[0].name
        self.preprocessing_config = {
            "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
            "std": np.array([0.229, 0.224, 0.225], dtype=np.float32),
        }

    def __get_input_size(self, config):
        return config["model_config"]["image_size"]

    def load_config(self, config_path):
        with open(config_path, "r", encoding="utf8") as f:
            config = json.load(f)
        return config

    def __init_model(self):
        assert self.device in ["cpu", "gpu"], "{} not in allowed device list".format(
            self.device
        )
        sess_options = rt.SessionOptions()

        # Set graph optimization level
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = rt.InferenceSession(
            self.onnx_file,
            sess_options=sess_options,
            providers=[
                "CPUExecutionProvider"
                if self.device == "cpu"
                else "CUDAExecutionProvider"
            ],
        )

        return session

    def pre_processing(self, image, roi_area):
        """Preprocessing image:
        - Convert BGR image to RGB
        -

        Args:
            img (_type_): _description_
            imgsz (int, optional): _description_. Defaults to 128.

        Returns:
            _type_: _description_
        """
        ori_img = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(roi_area) > 0:
            xmin, ymin, xmax, ymax = roi_area[:4]
        else:
            xmin, ymin, xmax, ymax = 0, 0, image.shape[0], image.shape[1]

        xmin, ymin, xmax, ymax = list(map(int, [xmin, ymin, xmax, ymax]))
        xmin, xmax = np.clip([xmin, xmax], 0, image.shape[1])
        ymin, ymax = np.clip([ymin, ymax], 0, image.shape[0])
        image = image[ymin:ymax, xmin:xmax, :]
        image, preprocessed_params = utils.resize_and_padding(
            image, self.input_size[:2]
        )        
        
        image = utils.normalize_image(
            image,
            mean=self.preprocessing_config["mean"],
            std=self.preprocessing_config["std"],
        )
        image = np.expand_dims(image, 0)
        return image, preprocessed_params, [xmin, ymin, xmax, ymax], ori_img

    def post_processing(
        self,
        image_origin,
        output,
        preprocess_param,
        roi_area,
        is_get_image_result,
        default_threshold=0.5,
        prob_dict={},
    ):
        """post processing segmentation output

        Args:
            image_origin (np.ndarray): _description_
            output (np.ndarray): _description_
            preprocess_param (_type_): _description_
            roi_area (_type_): _description_
            is_get_image_result (bool): _description_
            default_threshold (float, optional): _description_. Defaults to 0.5.
            prob_dict (dict, optional): _description_. Defaults to {}.

        Returns:
            _type_: _description_
        """
        output = output[0]

        if prob_dict:
            list_probability_threshold = utils.create_list_threshold(
                self.class_names, prob_dict, default_threshold
            )
        else:
            list_probability_threshold = [
                default_threshold for _ in range(len(self.class_names))
            ]

        prob = output[0]  # hxwxclass
        
        mask = np.argmax(prob, 2)  # h,w - get class label for each pixel
             
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                tmp = mask[i, j]
                if tmp == 0:
                    continue
                elif prob[i, j, tmp] < list_probability_threshold[tmp]:
                    mask[i, j] = 0

        ratio, (pad_w, pad_h), (original_width, original_height) = preprocess_param
        h, w = mask.shape[:2]
        mask = cv2.resize(
            mask[: (h - pad_h), : (w - pad_w)],
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        xmin, ymin, xmax, ymax = roi_area

        mask_origin = np.zeros(image_origin.shape[:2], dtype=np.uint8)
        mask_origin[ymin:ymax, xmin:xmax] = mask

        if is_get_image_result == False:
            mode = "test"
        else:
            mode = "inference"
        result_coco = utils.convert_mask_to_coco_annotation_format(
            image_origin,
            mask_origin,
            class_names=self.class_names,
            image_name="demo_python.jpg",
            mode=mode,
            folder_to_save="examples/output_onnx_python"
        )
        return {"predictions": result_coco}

    def __call__(
        self, image, roi_area, threshold, threshold_dict={}, is_get_image_result=False
    ):
        """

        Args:
            image (np.ndarray): cv2 image (BGR)
        """

        # print("Check model:")
        # input_tensor = np.load('input.npy')
        # print("Input tensor shape: ", input_tensor.shape)  #1*320*320*3
        # print("Input type: ", input_tensor.dtype)
        # print("Input tensor: ", input_tensor[0][0][:10])
        # output = self.model.infer_new_request({0: input_tensor})
        # print(output)

        input_tensor, preprocess_param, roi_area, ori_img = self.pre_processing(
            image, roi_area
        )
        output = self.model.run(None, {self.input_name: input_tensor})

        result = self.post_processing(
            image_origin=ori_img,
            output=output,
            preprocess_param=preprocess_param,
            roi_area=roi_area,
            default_threshold=threshold,
            is_get_image_result=is_get_image_result,
            prob_dict=threshold_dict,
        )
        return result

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model",
        help="path to onnx model",
        type=str,
        default="weights/onnx/model_2023110_191736_351410.onnx",
    )

    parser.add_argument(
        "--image",
        help="path to image",
        type=str,
        default="./examples/imgs/demo.jpg",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    infer = OnnxInfer(
        onnx_file=args.model,         
        device="CPU", 
    )

    image = cv2.imread(args.image)

    for i in range(1):

        start_time = time.time()
        result = infer(
            image,
            roi_area=[0, 0, 11900.1, 11900.2],
            threshold=0.1,
            threshold_dict={"hole": 0.6, "kizu": 0.4},
            is_get_image_result=True,
        )        
        print("process time: {}".format(time.time() - start_time))        
