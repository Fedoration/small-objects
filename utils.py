import time
from typing import Optional
import cv2
import copy
from pathlib import Path
import numpy as np


class Colors:
    def __init__(self):
        hex = (
            "FF3838",
            "2C99A8",
            "FF701F",
            "6473FF",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "FF9D97",
            "00C2FF",
            "344593",
            "FFB21D",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex_to_rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, ind, bgr: bool = False):
        """
        Convert an index to a color code.

        Args:
            ind (int): The index to convert.
            bgr (bool, optional): Whether to return the color code in BGR format. Defaults to False.

        Returns:
            tuple: The color code in RGB or BGR format, depending on the value of `bgr`.
        """
        color_codes = self.palette[int(ind) % self.n]
        return (color_codes[2], color_codes[1], color_codes[0]) if bgr else color_codes

    @staticmethod
    def hex_to_rgb(hex_code):
        """
        Converts a hexadecimal color code to RGB format.

        Args:
            hex_code (str): The hexadecimal color code to convert.

        Returns:
            tuple: A tuple representing the RGB values in the order (R, G, B).
        """
        rgb = []
        for i in (0, 2, 4):
            rgb.append(int(hex_code[1 + i : 1 + i + 2], 16))
        return tuple(rgb)


def apply_color_mask(image: np.ndarray, color: tuple):
    """
    Applies color mask to given input image.

    Args:
        image (np.ndarray): The input image to apply the color mask to.
        color (tuple): The RGB color tuple to use for the mask.

    Returns:
        np.ndarray: The resulting image with the applied color mask.
    """
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    (r[image == 1], g[image == 1], b[image == 1]) = color
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask


def visualize_object_predictions(
    image: np.array,
    object_prediction_list,
    rect_th: int = None,
    text_size: float = None,
    text_th: float = None,
    color: tuple = None,
    hide_labels: bool = False,
    hide_conf: bool = False,
    output_dir: Optional[str] = None,
    file_name: str = "prediction_visual",
    export_format: str = "png",
):
    """
    Visualizes prediction category names, bounding boxes over the source image
    and exports it to output folder.

    Args:
        object_prediction_list: a list of prediction.ObjectPrediction
        rect_th: rectangle thickness
        text_size: size of the category name over box
        text_th: text thickness
        color: annotation color in the form: (0, 255, 0)
        hide_labels: hide labels
        hide_conf: hide confidence
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select predefined classwise color palette if not specified
    if color is None:
        colors = Colors()
    else:
        colors = None
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3

    # add masks to image if present
    for object_prediction in object_prediction_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        # visualize masks if present
        if object_prediction.mask is not None:
            # deepcopy mask so that original is not altered
            mask = object_prediction.mask.bool_mask
            # set color
            if colors is not None:
                color = colors(object_prediction.category.id)
            # draw mask
            rgb_mask = apply_color_mask(mask, color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.6, 0)

    # add bboxes to image if present
    for object_prediction in object_prediction_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()

        bbox = object_prediction.bbox.to_xyxy()
        category_name = object_prediction.category.name
        score = object_prediction.score.value

        # set color
        if colors is not None:
            color = colors(object_prediction.category.id)
        # set bbox points
        point1, point2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        # visualize boxes
        cv2.rectangle(
            image,
            point1,
            point2,
            color=color,
            thickness=rect_th,
        )

        if not hide_labels:
            # arange bounding box text location
            label = f"{category_name}"

            if not hide_conf:
                label += f" {score:.2f}"

            box_width, box_height = cv2.getTextSize(
                label, 0, fontScale=text_size, thickness=text_th
            )[
                0
            ]  # label width, height
            outside = point1[1] - box_height - 3 >= 0  # label fits outside box
            point2 = point1[0] + box_width, (
                point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
            )
            # add bounding box text
            cv2.rectangle(image, point1, point2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                image,
                label,
                (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
                0,
                text_size,
                (255, 255, 255),
                thickness=text_th,
            )

    # export if output_dir is present
    if output_dir is not None:
        # export image with predictions
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = str(Path(output_dir) / (file_name + "." + export_format))
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}
