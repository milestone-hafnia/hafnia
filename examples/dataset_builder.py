# This should be moved to mdi-cli or a separate package
import collections

## from prettytable import PrettyTable
import inspect
import json
from abc import ABCMeta, abstractmethod
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import datasets
from datasets import ClassLabel, DatasetDict

import mdi

# dataset = mdi.load_dataset("mnist")
# dataset = mdi.load_dataset("[ADD SOME CUSTOM DATASET SPECIFICATION]")
CLASS_NAME = "class_name"
CLASS_IDX = "class_idx"


def capture_init_args(init_func: callable) -> callable:
    @wraps(init_func)
    def wrapper(self: DatasetTransform, *args: Tuple, **kwargs: Dict) -> None:
        args_with_self = inspect.signature(init_func).bind(self, *args, **kwargs).arguments
        # Capture the arguments passed to the __init__ method
        self.set_init_args(args_with_self)

        # Call the original __init__ method
        return init_func(self, *args, **kwargs)

    return wrapper


class CaptureInitArgsMixin:
    _init_args = None

    def is_type(self, config_dict: Dict) -> bool:
        return self.__class__.__name__ == config_dict["__type__"]

    def set_init_args(self, init_args: Dict) -> None:
        self._init_args = init_args

    def get_init_args(self) -> Dict:
        if self._init_args is None:
            raise ValueError(
                "Init arguments haven't been captured. Use '@capture_init_args decorator' on the __init__ method"
            )
        init_args = dict(self._init_args)
        class_type = init_args.pop("self")

        # Add '__type__' first for readability
        init_args = {"__type__": class_type.__class__.__name__, **init_args}
        return init_args

    def to_config(self) -> Dict:
        init_args = self.get_init_args()

        return init_args

    @classmethod
    def check_config(cls, config_dict: Dict) -> None:
        if "__type__" not in config_dict:
            raise ValueError("Config dictionary should have '__type__' key")
        config_dict = config_dict.copy()
        transform_type = config_dict.pop("__type__")
        if not cls.__name__ == transform_type:
            raise ValueError(
                f"Config dictionary is for a different transform type: {transform_type}"
            )
        return config_dict

    @classmethod
    def from_config(cls, config_dict: Dict) -> "DatasetTransform":
        config_dict = cls.check_config(config_dict)
        return cls(**config_dict)


# Define an abstract base class
class DatasetTransform(CaptureInitArgsMixin, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        pass


class DatasetBuilder(CaptureInitArgsMixin, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self) -> DatasetDict:
        pass


def dataset_builder_from_specs(
    cfg: str | tuple | list | DatasetBuilder,
) -> DatasetBuilder:
    if isinstance(cfg, str):
        return MdiLoadDataset(cfg)  # We could make a auto-detection an appropriate loading function
    if isinstance(cfg, tuple):
        return Concatenate(*cfg)
    if isinstance(cfg, list):
        return LoadDatasetWithTransforms(dataset=cfg[0], transforms=cfg[1:])
    if isinstance(cfg, DatasetBuilder):
        return cfg
    raise ValueError(f"Invalid config type: {cfg}")


def creator_from_config(cfg: Dict) -> DatasetBuilder:
    if "__type__" not in cfg:
        return cfg
    type_creator = cfg["__type__"]
    if type_creator == "LoadDatasetWithTransforms":
        return LoadDatasetWithTransforms.from_config(cfg)
    if type_creator == "Concatenate":
        return Concatenate.from_config(cfg)
    if type_creator == "MdiLoadDataset":
        return MdiLoadDataset.from_config(cfg)
    if type_creator == "LoadFromDisk":
        return LoadFromDisk.from_config(cfg)
    if type_creator == "MapObjectClassNames":
        return MapObjectClassNames.from_config(cfg)
    if type_creator == "MapClassNames":
        return MapClassNames.from_config(cfg)
    raise ValueError(f"Unknown creator/transform type: {type_creator}")


class LoadDatasetWithTransforms(DatasetBuilder):  # This is generated from a list
    @capture_init_args
    def __init__(
        self,  # Inputs are a dataset and transforms. Should it simply be a list?
        dataset: str | DatasetBuilder | List | Tuple,
        transforms: List[DatasetTransform],
    ) -> None:
        self.dataset_creator: DatasetBuilder = dataset_builder_from_specs(dataset)
        assert isinstance(transforms, list), "Transforms should be a list"
        self.transforms = transforms

    def to_config(self) -> Dict:
        transforms = [transform.to_config() for transform in self.transforms]
        return {
            "__type__": self.__class__.__name__,
            "dataset": self.dataset_creator.to_config(),
            "transforms": transforms,
        }

    @classmethod
    def from_config(cls, config_dict: Dict) -> "LoadDatasetWithTransforms":
        config_dict = cls.check_config(config_dict)
        dataset = creator_from_config(config_dict["dataset"])
        transforms = [
            creator_from_config(transform_config) for transform_config in config_dict["transforms"]
        ]
        return cls(dataset, transforms)

    def __call__(self) -> DatasetDict:
        dataset = self.dataset_creator()
        for transform in self.transforms:
            dataset = transform(dataset)
        return dataset


class Concatenate(DatasetBuilder):  # This is generated from a tuple
    @capture_init_args
    def __init__(
        self,
        *creators: str | DatasetBuilder,
    ) -> None:
        self.creators: List[DatasetBuilder] = [dataset_builder_from_specs(cfg) for cfg in creators]

    def to_config(self) -> Dict:
        creators = [dataset.to_config() for dataset in self.creators]
        return {"__type__": self.__class__.__name__, "creators": creators}

    @classmethod
    def from_config(cls, config_dict: Dict) -> "Concatenate":
        config_dict = cls.check_config(config_dict)
        creators = [
            creator_from_config(creator_config) for creator_config in config_dict["creators"]
        ]
        return cls(*creators)

    def __call__(self) -> DatasetDict:
        datasets_list: List[DatasetDict] = [creator() for creator in self.creators]
        dataset_splits_list = collections.defaultdict(list)
        for dataset in datasets_list:
            for split_name, split_dataset in dataset.items():
                dataset_splits_list[split_name].append(split_dataset)

        dataset_splits = {}
        for split_name, datasets_in_split in dataset_splits_list.items():
            dataset_splits[split_name] = datasets.concatenate_datasets(datasets_in_split)

        return datasets.DatasetDict(dataset_splits)


class MdiLoadDataset(DatasetBuilder):
    @capture_init_args
    def __init__(self, name: str, force_redownload: bool = False, verbose: bool = False) -> None:
        self.name = name
        self.force_redownload = force_redownload
        self.verbose = verbose

    def __call__(self) -> DatasetDict:
        return mdi.load_dataset(
            self.name, force_redownload=self.force_redownload, verbose=self.verbose
        )


class LoadFromDisk(DatasetBuilder):
    @capture_init_args
    def __init__(
        self,
        dataset_path: str | Path,
        keep_in_memory: Optional[None] = None,
        storage_options: Optional[Dict] = None,
    ) -> None:
        self.dataset_path = str(dataset_path)
        self.keep_in_memory = keep_in_memory
        self.storage_options = storage_options

    def __call__(self) -> DatasetDict:
        return datasets.load_from_disk(
            dataset_path=self.dataset_path,
            keep_in_memory=self.keep_in_memory,
            storage_options=self.storage_options,
        )


class MapObjectClassNames(DatasetTransform):
    @capture_init_args
    def __init__(self, complete_class_mapping: Dict[str, str]):
        # Complete class mapping for the dataset means:
        # 1) If a class is not in the mapping, it will be removed from the dataset
        # 2) If a class is in the mapping, it will be remapped to the new class name
        # 3) The order of the classes (values in the dictionary) will be used to determine class ordering and class index
        # 4) Should support taking all lower level classes and mapping them to a higher level class
        #    e.g. "Vehicle" -> "vehicle", meaning all vehicle classes (such as Vehicle.Car, Vehicle.Bus, ...) will be mapped to "vehicle"
        # 5) Consider also explicit-options where removing classes is required to be listed.
        #    To remove "bag", a user would have to add '{"bag": "__remove__"}', to remove the class "bag" from the dataset or it will raise an error
        self.complete_class_mapping = complete_class_mapping
        self.class_names = remove_duplicates_keep_order(complete_class_mapping.values())

    def _map_class_name(self, class_name: str) -> str:
        return self.complete_class_mapping[class_name]

    def _map_old_class_name_to_new_class_idx(self, class_name: str) -> int:
        return self.class_names.index(self._map_class_name(class_name))

    def _transform_sample(self, sample: Dict[str, Any], org_class_names: List[str]) -> Dict:
        assert "objects" in sample, "Sample should have 'objects' feature"
        objects = sample["objects"]
        assert "category" in objects, "Objects should have 'bbox' feature"

        object_keys = objects.keys()
        object_list = [dict(zip(objects.keys(), values)) for values in zip(*objects.values())]

        keep_objects = []
        for obj in object_list:
            org_class_name = org_class_names[obj["category"]]
            if org_class_name not in self.complete_class_mapping:
                continue
            new_class_name = self.complete_class_mapping[org_class_name]
            obj["category"] = self.class_names.index(new_class_name)
            keep_objects.append(obj)

        sample["objects"] = {key: [obj[key] for obj in keep_objects] for key in object_keys}
        return sample

    def _transform_dataset_metadata(self, dataset: datasets.Dataset) -> datasets.Dataset:
        new_features = dataset.features.copy()
        new_features["objects"].feature["category"] = ClassLabel(names=self.class_names)
        dataset_new = dataset.cast(new_features)
        return dataset_new

    def __call__(self, dataset_splits: DatasetDict) -> DatasetDict:
        dataset_train = next(iter(dataset_splits))
        org_class_names = dataset_train.features["objects"].feature["category"].names

        def map_function(sample: Dict[str, Any]) -> Dict[str, Any]:
            return self._transform_sample(sample, org_class_names)

        dataset_updated = dataset_splits.map(map_function)
        return self._transform_dataset_metadata(dataset_updated)


class MapClassNames(DatasetTransform):  # Image classification
    @capture_init_args
    def __init__(
        self, complete_class_mapping: Dict[str, str], label_column_name: str = "label"
    ) -> None:
        self.class_mapping = complete_class_mapping
        self.label_name = label_column_name

    def __call__(self, dataset_splits: DatasetDict) -> DatasetDict:
        new_label_names = remove_duplicates_keep_order(self.class_mapping.values())
        new_class_label = ClassLabel(names=new_label_names)
        some_split_name = next(iter(dataset_splits))
        dataset_split = dataset_splits[some_split_name]
        class_label = dataset_split.features[self.label_name]

        class_mapper = create_class_mapper(
            self.class_mapping,
            class_label=class_label,
        )

        for split_name in dataset_splits:
            dataset_split = dataset_splits[split_name].map(class_mapper)
            dataset_split = dataset_split.cast_column(self.label_name, new_class_label)
            dataset_splits[split_name] = dataset_split
        return dataset_splits


def create_class_mapper(class_mapping: Dict[str, str], class_label: ClassLabel) -> Callable:
    old_index2name = {class_label.str2int(k): k for k in class_label.names}
    new_classes = remove_duplicates_keep_order(class_mapping.values())
    old_2_new_index_mapping = {
        old_index: new_classes.index(class_mapping[old_index2name[old_index]])
        for old_index in old_index2name
    }

    def map_label(sample: Dict) -> Dict:
        sample["label"] = old_2_new_index_mapping[sample["label"]]
        return sample

    return map_label


def apply_transforms_without_image_features(
    dataset: datasets.Dataset, transforms: List
) -> datasets.Dataset:
    """
    Removes images from the dataset, applies the transforms and then adds the images back to the dataset.
    This is simply an optimization to avoid decoding/reading of image features, when not needed
    """
    image_features = [k for k, v in dataset.features.items() if isinstance(v, datasets.Image)]
    dataset_with_images = dataset.select_columns(image_features)
    dataset_without_images = dataset.remove_columns(image_features)
    for transform in transforms:
        dataset_without_images = transform.transform_dataset(dataset_without_images)
    dataset = datasets.concatenate_datasets([dataset_without_images, dataset_with_images], axis=1)
    return dataset


def apply_decoding(dataset: datasets.Dataset, decoding: bool) -> datasets.Dataset:
    image_features = {k: v for k, v in dataset.features.items() if isinstance(v, datasets.Image)}
    for image_feature_name, image_feature in image_features.items():
        image_feature.decode = decoding
        dataset = dataset.cast_column(image_feature_name, datasets.Image(decode=decoding))
    return dataset


def remove_duplicates_keep_order(items: Iterable[str]) -> List[str]:
    """
    Remove duplicates from a list while keeping the order of the items

    >>> remove_duplicates_keep_order(["a", "b", "a", "c", "b"])
    ["a", "b", "c"]

    """
    return list(dict.fromkeys(items))


if __name__ == "__main__":
    # concatenate(LoadFromDisk("mnist0"), LoadFromDisk("mnist1"), LoadFromDisk("mnist2"))
    dataset_name = "mnist"

    path_data_mnist = Path(__file__).parents[1] / "data" / dataset_name
    if not path_data_mnist.exists():
        dataset = datasets.load_dataset(dataset_name)
        dataset.save_to_disk(str(path_data_mnist))

    creator_with_transform = LoadFromDisk(path_data_mnist)
    mnist = creator_with_transform()

    complete_class_mapping = {
        "0": "even",
        "1": "odd",
        "2": "even",
        "3": "odd",
        "4": "even",
        "5": "odd",
        "6": "even",
        "7": "odd",
        "8": "even",
        "9": "odd",
    }

    # Feature: Single dataset as a string
    dataset_creator = dataset_builder_from_specs("mnist")
    mnist_dataset = dataset_creator()

    # Feature: Explicit single dataset creator
    dataset_creator = dataset_builder_from_specs(LoadFromDisk(path_data_mnist))
    dataset_creator = dataset_builder_from_specs(MdiLoadDataset(path_data_mnist))

    # Feature: Transform datasets with lists
    class_mapping = MapClassNames(complete_class_mapping)
    creator_with_transform = dataset_builder_from_specs(["mnist", class_mapping])
    mnist_odd_even = creator_with_transform()

    # Feature: Merge datasets using tuples
    creator_merged: DatasetBuilder = dataset_builder_from_specs(("mnist", "mnist"))
    mnist_merged = creator_merged()

    # Feature: Nesting lists and tuples infinitely!
    creator_nested: DatasetBuilder = dataset_builder_from_specs(
        [("mnist", ("mnist", "mnist")), class_mapping]
    )
    mnist_nested = creator_nested()
    mnist_nested["train"].features
    sample = mnist_nested["train"][0]

    # Feature: To config and back (to create in backend?)
    creator_config = creator_nested.to_config()
    print(json.dumps(creator_config, indent=4))
    creator_again = creator_from_config(creator_config)

    # Feature: Transformations can also be executed one-by-one for debugging
    dataset = LoadFromDisk(path_data_mnist)()
    dataset = class_mapping(dataset)

    # Feature: Create convenient dataset builder specifications for our users
    coco2pascal_mapping = MapObjectClassNames(complete_class_mapping={})
    MSCOCO_2_PASCAL_VOC = ["mscoco", coco2pascal_mapping]
    PASCAL_EXTENDED = ("pascal", MSCOCO_2_PASCAL_VOC)

    pascal2person = MapObjectClassNames(complete_class_mapping={})
    mscoco2person = MapObjectClassNames(complete_class_mapping={})
    PERSON_DATASET = (["pascal", pascal2person], ["mscoco", mscoco2person])

    # All below items should be called with 'mdi.load_dataset'. E.g. mdi.load_dataset(("mnist", "mnist"))'
    DATASET_BUILDER_SPECIFICATIONS = [
        # Str: Pass name of the dataset as normally
        "mnist",
        # Tuple to concatenate datasets
        ("mnist", "mnist"),
        # List to create and transform dataset
        ["mnist", class_mapping],
        # Nested list and tuples (The feature nobody will use)
        ("mnist", ("mnist", ["mnist", class_mapping])),
        # Use explicit form (equivalent to above)
        Concatenate(
            MdiLoadDataset("mnist"),
            Concatenate(
                MdiLoadDataset("mnist"),
                LoadDatasetWithTransforms(MdiLoadDataset("mnist"), [class_mapping]),
            ),
        ),
        # Combining different types of dataset creators
        LoadDatasetWithTransforms("mnist", [class_mapping, class_mapping]),
    ]

    # Features to and from config
    for dataset_builder_specs in DATASET_BUILDER_SPECIFICATIONS:
        print(f"Creating dataset from config: \n{dataset_builder_specs}")
        creator_with_transform = dataset_builder_from_specs(dataset_builder_specs)
        dataset_creator_as_cfg = creator_with_transform.to_config()
        # dataset = creator()
        print("As config: \n")
        print(json.dumps(dataset_creator_as_cfg, indent=4))
        # test = json.dumps(creator_as_cfg, indent=4)
        # assert test == json.dumps(creator_as_cfg, indent=4)
        print()

        creator_again = creator_from_config(dataset_creator_as_cfg)
        creator_again_as_cfg = creator_again.to_config()

        assert dataset_creator_as_cfg == creator_again_as_cfg
        print()

        # def check(self, dataset: DatasetDict) -> DatasetDict:
        #     original_class_names = dataset.features[self.feature_name][0][CLASS_IDX].names
        #     renamed_org_classes = [
        #         name for name in self.complete_class_mapping if name in original_class_names
        #     ]
        #     removed_org_classes = [
        #         name for name in original_class_names if name not in renamed_org_classes
        #     ]

        #     added_classes = [
        #         v
        #         for k, v in self.complete_class_mapping.items()
        #         if k not in original_class_names
        #     ]

        #     list_class_transforms = []
        #     for original_class_name in original_class_names:
        #         str_class_transform = original_class_name
        #         if original_class_name in removed_org_classes:
        #             str_class_transform = str_class_transform + " (removed)"

        #         if original_class_name in renamed_org_classes:
        #             str_class_transform = (
        #                 str_class_transform
        #                 + f" -> {self._map_class_name(original_class_name)}"
        #             )

        #         list_class_transforms.append(str_class_transform)

        #     print("Class mappings: ")
        #     print("Renamed classes:")
        #     renaming_as_strings = [
        #         f"{name} -> {self.complete_class_mapping[name]}"
        #         for name in renamed_org_classes
        #     ]
        #     print(textwrap.indent("\n".join(renaming_as_strings), prefix="    "))
        #     print("Removed classes:")
        #     print(textwrap.indent("\n".join(removed_org_classes), prefix="    "))

        #     inverse_class_mapping = collections.defaultdict(list)
        #     for k, v in self.complete_class_mapping.items():
        #         inverse_class_mapping[v].append(k)
        #     # inverse_class_mapping_str = {k: ", ".join(mappings) for k, mappings in inverse_class_mapping.items()}

        #     print("")
        #     print("Classes after transformation:")
        #     table = PrettyTable()
        #     columns = ["Class Index", "Class Name", "Notes"]
        #     table.field_names = columns
        #     for column_name in columns:
        #         table.align[column_name] = "l"

        #     for i_class, class_name in enumerate(self.class_names):
        #         transforms = []
        #         from_class_name = inverse_class_mapping[class_name]
        #         if class_name in from_class_name:
        #             from_class_name.remove(class_name)

        #         if class_name in added_classes:
        #             transforms.append("Not in original dataset")

        #         if len(from_class_name) > 0:
        #             transform = f"Remapped from {', '.join(from_class_name)}"
        #             transforms.append(transform)

        #         transforms_as_string = ", ".join(transforms)
        #         table.add_row([i_class, class_name, transforms_as_string])

        #     print(table)
