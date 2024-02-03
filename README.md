# AnyToCocoConverter
This Python module provides a class, AnyToCocoConverter, which converts labels from the Anylabeling image annotation app to the Microsoft COCO format. This conversion is essential for machine learning applications that require annotations in the COCO format.

- [Anylabeling](https://anylabeling.nrl.ai/)  
- [COCO Dataset](https://cocodataset.org/)

### Features
The AnyToCocoConverter class includes several methods that facilitate the conversion process:
- seperate_annotation_files_from_image_files_in_directory: Separates image filenames from annotation filenames in a given directory.

- extract_polygons_from_annotaions_file: Extracts polygons from annotation files, removing unnecessary metadata.
- extract_categories: Extracts categories (labels) from the annotations and creates dictionaries for further processing.
- get_polygon_area: Calculates the area of a given polygon, which is required in the COCO dataset.
- convert_images_and_annotation_instances_to_coco_format: Converts images and annotation instances to the COCO format.
- create_coco: Creates the COCO dataset by copying images into a new directory and creating an annotations.json file.


### Installation

To install this module, you can clone the repository and import the AnyToCocoConverter class into your Python script.

### Usage
To use this module, you need to create an instance of the AnyToCocoConverter class, passing the root directory of your Anylabeling dataset as an argument. Then, you can call the create_coco method to create the COCO dataset. This method requires three arguments: info, licenses, and coco_root_dir, which are dictionaries and a path respectively.  
Here is a basic example:  

```python
from anylabeling_converter import AnyToCocoConverter

converter = AnyToCocoConverter(anylabeling_root_dir)
converter.create_coco(info, licenses, coco_root_dir)
```

### Contributing
Contributions to this project are welcome. Please feel free to open an issue or submit a pull request.
### License
This project is licensed under the MIT License. Please see the LICENSE file for more details.
### Contact
If you have any questions or feedback, please feel free to contact me.
### Acknowledgements
I would like to thank the developers of the Anylabeling app for providing a robust platform for image annotation