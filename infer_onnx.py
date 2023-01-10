import argparse
import os
import cv2
import numpy as np
import logging as log
import sys
import json
import utils
import matplotlib.pyplot as plt
import time

import onnxruntime as rt

log.basicConfig(
    format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout
)


class OpenVinoInfer:
    def __init__(self, onnx_file, model_config=None, device="CPU"):
        """

        Args:
            model_path (str): model .xml file
            input_size (int): with = height
            model_config (str): model config .json
        """
        self.onnx_file = onnx_file
        self.device = device.lower()

        self.config = self.load_config(config_path=model_config)
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
        print("image resize: ", image[0][:10])
        cv2.imwrite("image_resized_py.jpg", image)
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

        # output = next(iter(output.values()))
        # print(output[0].shape)
        output = output[0]

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

        print("Output prob shape: ", prob.shape)
        print("Output prob: ", prob)

        mask = np.argmax(prob, 2)  # h,w - get class label for each pixel
        print(mask)
        print("Mask first output: ", (mask != 0).sum())
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
            image_name="default.jpg",
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

        
        # input_tensor = np.random.randn(1, 320, 320, 3)
        # print("Input tensor shape: ", input_tensor.shape)  #1*320*320*3
        print("Input type: ", input_tensor.dtype)
        print("Input tensor: ", input_tensor)
        
        output = self.model.run(None, {self.input_name: input_tensor})
        print(output)

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
        default="weights/onnx/checkpoint_segmentation.onnx",
    )
    parser.add_argument(
        "--config",
        help="path to json config of h5 model",
        type=str,
        default="/mnt/Sources/vimage_v1/script_backend/checkpoint_anomaly/best_line_1.json",
    )

    parser.add_argument(
        "--image",
        help="path to model config .json",
        type=str,
        default="/home/dungdv/Downloads/Telegram Desktop/2P_320/NG/NG19.jpg",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer = OpenVinoInfer(
        onnx_file=args.model, 
        model_config=args.config, 
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
        print(result)
        # mask_new, contours, hierarchy = cv2.findContours(np.asarray(masks), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ### VISUALIZE RESULT ###
        # segment = result[1]['segmentation']
        # bbox = result[1]['bbox']
        # heatmap = result[1]['heatmap']

        # image_bin = np.zeros_like(image)
        # contours = [np.array(seg).reshape((-1, 2)) for seg in segment]
        # cv2.drawContours(image_bin, np.array(contours), -1, (255, 255, 255), 3)

        # image_draw_box = image.copy()
        # [cv2.rectangle(image_draw_box, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1) for box in bbox]

        # num_cols = 4
        # figure_size = (num_cols * 5, 5)
        # figure, axis = plt.subplots(1, num_cols, figsize=figure_size)
        # figure.subplots_adjust(right=0.9)
        # axis[0].imshow(image, vmin=0, vmax=255)
        # axis[0].title.set_text("original image")

        # axis[1].imshow(heatmap, cmap='viridis')
        # axis[1].title.set_text("Predicted Heat Map")

        # axis[2].imshow(image_bin, cmap='gray', vmin=0, vmax=255)
        # axis[2].title.set_text("Predicted mask")

        # axis[3].imshow(image_draw_box, vmin=0, vmax=255)
        # axis[3].title.set_text("Predicted bbox")

        # figure.canvas.draw()
        # plt.savefig("img.jpg")

        print("process time: {}".format(time.time() - start_time))
        # break
