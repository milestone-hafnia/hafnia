from __future__ import annotations

from typing import List, Type

from .bbox import Bbox
from .bitmask import Bitmask
from .classification import Classification
from .polygon import Polygon
from .primitive import Primitive
from .segmentation import Segmentation
from .utils import (
    class_color_by_name,
)

PRIMITIVE_TYPES: List[Type[Primitive]] = [Bbox, Classification, Polygon, Bitmask]
PRIMITIVE_NAME_TO_TYPE = {cls.__name__: cls for cls in PRIMITIVE_TYPES}
PRIMITIVE_COLUMN_NAMES: List[str] = [PrimitiveType.column_name() for PrimitiveType in PRIMITIVE_TYPES]
