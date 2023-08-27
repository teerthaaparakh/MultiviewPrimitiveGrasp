import numpy as np

STICK_LEN = 0.08
STICK_RADIUS = 0.0005
CANONICAL_LEN = 0.15

NORMALIZATION_CONST = 500

OBJECT_DICTS = {
    "semi_sphere": 0,
    "sphere": 1,
    "cuboid": 2,
    "ring": 3,
    "cylinder": 4,
    "stick": 5,
}

NUM_TRAINING_DATA = 1000
NUM_TEST_DATA = 10

NUM_BINS = 9

HM_WT = 0.5
WD_WT = 0.5

NUM_TOPS = 100
