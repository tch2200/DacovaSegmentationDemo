import os, cv2
import numpy as np

def resize_and_padding(image, target_size):
    target_width, target_height = target_size
    height, width = image.shape[:2]
    scale_width, scale_height = target_width/width, target_height/height

    if scale_width < scale_height:
        new_height = int(scale_width * height)
        image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_NEAREST)
    else:
        new_width = int(scale_height * width)
        image = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_NEAREST)
    new_height, new_width = image.shape[:2]
    pad = (target_width-new_width, target_height-new_height)
    ratio = min(scale_width, scale_height)
    image = cv2.copyMakeBorder(image, 0, pad[1], 0, pad[0], cv2.BORDER_CONSTANT, None, value = 0)

    return image, (ratio, pad, (width, height))

def normalize_image(image, mean, std):
    image = image.astype(np.float32)
    image = image / 255.0
    image = (image - mean) / std
    return image

def convert_mask_to_coco_annotation_format(
        img_origin, 
        mask, 
        class_names,
        image_id=1, 
        image_name="demo_python.jpg",
        mode="inference",
        threshold_area=1, 
        threshold_width=1, 
        threshold_height=1
    ):        
        mask_list = []
        for i in range(1, len(class_names)):
            mask_list.append(
                np.where(mask==i, 255, 0)
            )
        
        labels_info = []
        for idx, mask in enumerate(mask_list):
            # opencv 3.2
            contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            bbox = []
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, 0.02 * peri, True)
                flatten_contour = contour.flatten().tolist()
                if len(flatten_contour) > 4:
                    area = cv2.contourArea(contour)
                    if area < threshold_area:
                        continue
                    x,y,w,h = cv2.boundingRect(contour)
                    if w < threshold_width or h < threshold_height:
                        continue
                    xmin, xmax = np.clip([x, x+w], 0, mask.shape[1])
                    ymin, ymax = np.clip([y, y+h], 0, mask.shape[0])
                    bbox.append([
                        int(xmin), int(ymin), int(xmax), int(ymax)
                    ])
                    segmentation.append(flatten_contour)
            if len(segmentation) == 0:
                continue
            labels_info.append({
                "segmentation": segmentation,  # list of poly
                "bbox": bbox,
                "category_id": idx+1,
                "category": class_names[idx+1],
                "image_id": image_id
            })

        print("Label info: ", labels_info)

        if mode == "inference":
            result_image = visualize_result_on_image(img_origin, labels_info)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)            

            folder_to_save = "examples/output_openvino_python"
            os.makedirs(folder_to_save, exist_ok=True)
            path_to_save = os.path.join(folder_to_save, image_name)
            cv2.imwrite(path_to_save, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            return [{
                "image_id": image_id,
                "result_image": os.path.realpath(path_to_save)
            }]

        return labels_info

def visualize_result_on_image(image, result_coco):
    """
        add mask and image and draw bbox
    """
    image_show = image.copy()
    for idx, label in enumerate(result_coco):
        segmentation = label['segmentation']
        bbox = label['bbox']
        category_name = label['category']
        for seg, box in zip(segmentation, bbox):
            image_draw = np.zeros(image.shape, dtype=np.uint8)
            cv2.fillPoly(image_draw, [np.array(seg).reshape(-1, 2)], (255, 255, 255))
            image_show = cv2.addWeighted(image_show, 0.8, image_draw, 0.2, 0)
            cv2.rectangle(image_show, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            cv2.putText(image_show, category_name, (box[0], max(box[1]-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 0, 0), 1, cv2.LINE_AA)
    return image_show

def create_list_threshold(class_names, prob_dict, prob_threshold):
        list_probs = [prob_threshold for i in range(0, len(class_names))]
        for name in prob_dict.keys():
            assert name in class_names
            idx = class_names.index(name)
            list_probs[idx] = prob_dict[name]
        return list_probs