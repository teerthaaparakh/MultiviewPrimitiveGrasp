import sys
import os, os.path as osp

sys.path.append(os.getenv("KGN_DIR"))
from dataloader.dataset_func import dataset_function
from dataloader.dataset_func_vae import dataset_function_vae
from utils.other_configs import NUM_TRAINING_DATA, NUM_TEST_DATA
import pickle
from utils.path_util import get_pickled_dataset


result_training_1 = dataset_function(NUM_TRAINING_DATA)
with open(osp.join(get_pickled_dataset(), "sim_train.pkl"), 'wb') as f:
    pickle.dump(result_training_1, f)

result_training_2 = dataset_function(NUM_TEST_DATA)
with open(osp.join(get_pickled_dataset(), "sim_test.pkl"), 'wb') as f:
    pickle.dump(result_training_2, f)

result_training_3 = dataset_function_vae(NUM_TRAINING_DATA)
with open(osp.join(get_pickled_dataset(), "sim_train_vae.pkl"), 'wb') as f:
    pickle.dump(result_training_3, f)

result_training_4 = dataset_function_vae(NUM_TEST_DATA)
with open(osp.join(get_pickled_dataset(), "sim_test_vae.pkl"), 'wb') as f:
    pickle.dump(result_training_4, f)

