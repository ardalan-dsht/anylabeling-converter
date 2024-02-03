import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


class AnyToCocoConverter:
    def __init__(self, anylabeling_root_dir: Path) -> None:
        self.anylabeling_root_dir = anylabeling_root_dir
        self.filenames = self.seperate_annotation_files_from_image_files_in_directory()
        self.image_filename_with_annotations = (
            self.extract_polygons_from_annotaions_file()
        )
        self.categories = self.extract_categories()
        (
            self.coco_annotations,
            self.coco_images,
        ) = self.convert_images_and_annotation_instances_to_coco_format()

    def seperate_annotation_files_from_image_files_in_directory(
        self,
    ) -> Dict[str, list[str]]:
        """
        Anylabeling app stores all images in a directory with a json file for each of them,
        the json file stores annotations and shapes inside these files for each image.
        This method iterates over all filenames in directory and separates image filenames with annotation
        filenames.

        Returns:
            Dict[str, list[str]]: Image filenames and annotation filenames are stored in two different lists,
            these lists are values of two keys which this method returns.
        """
        all_image_filenames = []
        all_annotation_filenames = []
        for filename in os.listdir(self.anylabeling_root_dir):
            if filename.split(".")[-1] == "json":
                all_annotation_filenames.append(filename)
            elif filename.split(".")[-1] in [
                "jpg",
                "jpeg",
                "JPEG",
                "JPG",
                "PNG",
                "png",
            ]:
                all_image_filenames.append(filename)
            else:
                print(f"This is not a valid file : {filename}")
        all_annotation_filenames = sorted(all_annotation_filenames)
        all_image_filenames = sorted(all_image_filenames)
        filenames = {
            "image_filenames": all_image_filenames,
            "annotation_filenames": all_annotation_filenames,
        }
        return filenames

    def extract_polygons_from_annotaions_file(
        self,
    ) -> [str, Dict[str, List[int] | int]]:
        """
        Since we don't need the apps metadata stored on annotation files we can just remove them.
        Only stores useable info which are image height, width with annotation polygon.

        Returns:
            [str,Dict[str, List[int] | int]]: Image filenames with annotations(polygons) with image height, width.
        """
        image_name_with_all_annotations = {}
        for filename in self.filenames["annotation_filenames"]:
            # Read annotations file.
            with open(f"{self.anylabeling_root_dir}/{filename}", "r") as json_file:
                values_in_annotation_file = json.load(json_file)
            shapes = values_in_annotation_file["shapes"]
            # Filter annotation files which have no labels stored in them.
            if len(shapes) != 0:
                # Store all annotations for image into a list.
                annotations_for_image = []
                for each_shape in shapes:
                    # Only extract polygons
                    if each_shape["shape_type"] == "polygon":
                        label = each_shape["label"]
                        points = each_shape["points"]
                        sample = {"label": label, "points": points}
                        annotations_for_image.append(sample)
                # Filenames for annotations and images are the same, only format is different.
                image_file_formats = ["jpg", "jpeg", "JPEG", "JPG", "PNG", "png"]
                for file_format in image_file_formats:
                    new_image_filename = filename.split(".")[0] + f".{file_format}"
                    if new_image_filename in self.filenames["image_filenames"]:
                        image_name_with_all_annotations[new_image_filename] = {
                            "annotations": annotations_for_image,
                            "image_height": values_in_annotation_file["imageHeight"],
                            "image_width": values_in_annotation_file["imageWidth"],
                        }
        return image_name_with_all_annotations

    def extract_categories(self) -> Dict[str, dict]:
        """
        For further processing you need to have a ground truth defined for your categories(labels),
        for example during the entire process and in your new coco dataset category ID must not change!
        It's a simpler solution to instead of inverting one main category dict over and over again just create
        different category dicts but with same values interchanged.

        Returns:
            Dict[str, dict]: This dictionary stores the different dictionaries needed during the process.
        """
        all_categories = []
        for image_filename in self.image_filename_with_annotations:
            all_annotations_for_filename = self.image_filename_with_annotations[
                image_filename
            ]["annotations"]
            for each_annotation in all_annotations_for_filename:
                all_categories.append(each_annotation["label"])
        all_categories = set(all_categories)
        index_to_category = {}
        category_to_index = {}
        for idx, category in enumerate(all_categories):
            index_to_category[idx] = category
            category_to_index[category] = idx
        categories_for_coco = []
        for idx, category in index_to_category.items():
            category_sample = {
                "id": idx,
                "name": category,
                "supercategory": "undefined",
            }
            categories_for_coco.append(category_sample)
        categories = {
            "categories_for_coco": categories_for_coco,
            "category_to_index": category_to_index,
            "index_to_category": index_to_category,
        }
        return categories

    @staticmethod
    def get_polygon_area(polygon: np.ndarray) -> int:
        """
        Calculates polygon area which is needed in coco dataset.

        Args:
            polygon (np.ndarray): Extracted from anylabeling json files.

        Returns:
            int: Area for given polygon.
        """
        x = polygon[:, 0]
        y = polygon[:, 1]
        s1 = np.sum(x * np.roll(y, -1))
        s2 = np.sum(y * np.roll(x, -1))
        area = 0.5 * np.absolute(s1 - s2)
        return area

    def convert_images_and_annotation_instances_to_coco_format(
        self,
    ) -> Tuple[list, list]:
        """
        The main convertion process.
        Images and Annotations in coco dataset are created in this method.

        Returns:
            Tuple[list, list]: The list values which are gonna be in annotation.json for coco dataset.
        """
        all_images_in_coco = []
        all_annotations_in_coco = []
        annotation_id = 0
        for image_id, image_filename in enumerate(self.filenames["image_filenames"]):
            image = Image.open(f"{self.anylabeling_root_dir}/{image_filename}")
            image = np.asarray(image)
            image_sample = {
                "id": image_id,
                "file_name": image_filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "date_captured": None,
                "license": 0,
            }
            all_images_in_coco.append(image_sample)
            if image_filename in self.image_filename_with_annotations:
                all_annotations_for_given_image = self.image_filename_with_annotations[
                    image_filename
                ]["annotations"]
                for annotation in all_annotations_for_given_image:
                    points = np.array(annotation["points"])
                    area = self.get_polygon_area(points)
                    xmin = int(np.min(points[:, 0]))
                    ymin = int(np.min(points[:, 1]))
                    xmax = int(np.max(points[:, 0]))
                    ymax = int(np.max(points[:, 1]))
                    box_width = xmax - xmin
                    box_height = ymax - ymin
                    bbox = [xmin, ymin, box_width, box_height]
                    segmentation = [points.flatten().tolist()]
                    annotation_sample = {
                        "image_id": image_id,
                        "id": annotation_id,
                        "category_id": self.categories["category_to_index"][
                            annotation["label"]
                        ],
                        "iscrowd": 0,
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "area": area,
                    }
                    all_annotations_in_coco.append(annotation_sample)
                    annotation_id += 1
        return all_annotations_in_coco, all_images_in_coco

    def create_coco(self, info: dict, licenses: list, coco_root_dir: Path) -> None:
        """
        Creates coco dataset by copying images into the new coco root directory and creating an annotations.json file

        Args:
            info (dict): _description_
            licenses (list): _description_
            coco_root_dir (Path): _description_
        """
        coco = {
            "info": info,
            "license": licenses,
            "categories": self.categories["categories_for_coco"],
            "images": self.coco_images,
            "annotations": self.coco_annotations,
        }
        for image_filename in self.filenames["image_filenames"]:
            shutil.copy(
                self.anylabeling_root_dir / image_filename,
                coco_root_dir / image_filename,
            )
        with open(coco_root_dir / Path("annotations.json"), "w+") as json_file:
            json.dump(coco, json_file)
