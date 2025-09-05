import numpy as np

COMMON_OBJECTS = [
    "chair",
    "couch",
    "potted plant",
    "bed",
    "toilet",
    "tv",
    "dining-table",
    "oven",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "cup",
    "bottle",
]


COLOR_MAP_FOR_OBSTACLE = np.array(
    [
        [255, 255, 255],  # free
        [242, 242, 242],  # explored
        [100, 100, 100],  # obstacle
    ],
    dtype=np.uint8,
)

COLOR_MAP_FOR_OBJECT = np.array(
    [
        [239, 199, 168],
        [239, 226, 168],
        [226, 239, 168],
        [199, 239, 168],
        [172, 239, 168],
        [168, 239, 190],
        [168, 239, 217],
        [168, 235, 239],
        [168, 208, 239],
        [168, 181, 239],
        [181, 168, 239],
        [208, 168, 239],
        [235, 168, 239],
        [239, 168, 217],
        [239, 168, 190],
        [239, 168, 160],
    ],
    dtype=np.uint8,
)
