import cv2
from config import LABEL_DICTS

def bbox_convert(results):
    bboxes = []
    for result in results:
        x_min = round(640 * (result["xcenter"] - result["width"] / 2))
        x_max = round(640 * (result["xcenter"] + result["width"] / 2))
        y_min = round(640 * (result["ycenter"] - result["height"] / 2))
        y_max = round(640 * (result["ycenter"] + result["height"] / 2))
        bbox_label = LABEL_DICTS[result["class"]]
        bbox_dict = {
            "bbox": [x_min, y_min, x_max, y_max],
            "label": bbox_label
        }
        bboxes.append(bbox_dict)
    return bboxes

# Draw the bounding box and the label
def draw_bbox(image, bboxes, path_dir: str):
    for i in bboxes:
        bbox = i["bbox"]
        label = i["label"]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(image, str(label), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save the image
    cv2.imwrite(path_dir, image)