import argparse
import cv2
import numpy as np

import json
import utils
import time
from pathlib import Path

from openvino.runtime import Core, Layout, Type
from openvino.preprocess import PrePostProcessor

class OpenVinoInfer:
    def __init__(self, model_path, model_config=None, device="CPU", layout="NHWC"):
        """
        Args:
            model_path (str): model .xml file
            input_size (int): with = height
            model_config (str): model config .json
        """
        config_path = Path(model_path).with_suffix(".json")

        self.config = self.load_config(config_path=config_path)
        
        self.input_size = self.__get_input_size(self.config)  # [h, w, c]
        assert len(self.input_size) == 3, "length of input size must be 3"
        self.layout = layout

        self.model = self.load_model(model_path, device)
        self.class_names = self.config["model_config"]["class_name"]

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

    def load_model(self, model_path, device):
        openvino_runtime_core = Core()
        model = openvino_runtime_core.read_model(model_path)

        # n, h, w, c = 1, self.input_size[0], self.input_size[1], self.input_size[2]

        ppp = PrePostProcessor(model)

        # 1) Set input tensor information:
        # - input() provides information about a single model input
        # - reuse precision and shape from already available `input_tensor`
        # - layout of data is 'NHWC'
        # ppp.input().tensor().set_shape([n, h, w, c]).set_element_type(
        #     Type.u8
        # ).set_layout(
        #     Layout(self.layout)
        # )  # noqa: ECE001, N400

        ppp.input().tensor().set_element_type(Type.f32).set_layout(Layout(self.layout)) 
        
        # 2) Adding explicit preprocessing steps:
        # - apply linear resize from tensor spatial dims to model spatial dims
        # ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

        # 3) Here we suppose model has 'NHWC' layout for input
        ppp.input().model().set_layout(Layout(self.layout))

        # 4) Set output tensor information:
        # - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(Type.f32)

        # 5) Apply preprocessing modifying the original 'model'
        model = ppp.build()

        compiled_model = openvino_runtime_core.compile_model(model, device)
        return compiled_model

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
        default_threshold=0.2,
        prob_dict={},
    ):

        output = next(iter(output.values()))

        if prob_dict:
            list_probability_threshold = utils.create_list_threshold(
                self.class_names, prob_dict, default_threshold
            )
        else:
            list_probability_threshold = [
                default_threshold for _ in range(len(self.class_names))
            ]
        print("list_probability_threshold: ", list_probability_threshold)
        prob = output[0]  # hxwxclass

        mask = np.argmax(prob, 2)  # h,w - get class label for each pixel
        
        # prob_max = np.max(prob, 2)
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
        )

        return {"predictions": result_coco}

    def __call__(
        self, image, roi_area, threshold, threshold_dict={}, is_get_image_result=False
    ):
        """
        Args:
            image (np.ndarray): cv2 image (BGR)
        """
        input_tensor, preprocess_param, roi_area, ori_img = self.pre_processing(
            image, roi_area
        )
        
        output = self.model.infer_new_request({0: input_tensor})        

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
        help="path to openvino .xml file",
        type=str,
        default="model_20221222_9229_664304.xml",
    )
    parser.add_argument(
        "--image",
        help="path to image",
        type=str,
        default="./examples/inputs/demo.jpg",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    infer = OpenVinoInfer(
        model_path=args.model,         
        device="CPU", 
        layout="NHWC"
    )

    image = cv2.imread(args.image)

    for i in range(1):

        start_time = time.time()
        result = infer(
            image,
            roi_area=[0, 0, 11900.1, 11900.2],
            threshold=0.5,
            threshold_dict={"hole": 0.6, "kizu": 0.4},
            is_get_image_result=True,
        )

        print("process time: {}".format(time.time() - start_time))
