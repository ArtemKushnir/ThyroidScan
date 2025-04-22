import shutil
from pathlib import Path


def split_files_by_type(data_dir: str, xml_dir: str = "xml_files", images_dir: str = "images") -> None:
    """
    Divides the directory with xml and jpg files into two directories.

    :param data_dir: directory with data.
    :param xml_dir: directory with xml files.
    :param images_dir: directory with images.
    """
    data_path = Path(data_dir)

    Path(xml_dir).mkdir(exist_ok=True)
    Path(images_dir).mkdir(exist_ok=True)

    for file in data_path.glob("*"):
        if file.suffix.lower() == ".xml":
            shutil.copy(str(file), str(Path(xml_dir) / file.name))
        elif file.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy(str(file), str(Path(images_dir) / file.name))


if __name__ == "__main__":
    data_dir = "data"
    xml_dir = "xml_files"
    images_dir = "images"

    split_files_by_type(data_dir, xml_dir, images_dir)
