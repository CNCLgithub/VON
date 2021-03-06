import sys
from os.path import join

from .test_model_simple import TestModelSimple
from .options import opt
import torch


def preprocess(opt, gpu_ids):
    # set gpu ids
    str_ids = gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])


def load_models(gpu_ids, objects_to_load):
    # preprocess opt
    preprocess(opt, gpu_ids)

    car_model = None
    chair_model = None
    airplane_model = None
    table_model = None

    model3D_dir = opt.model3D_dir

    # load car_model
    if "car" in objects_to_load:
        opt.model3D_dir = join(model3D_dir, "car_df")
        car_model = TestModelSimple(opt)

    # load chair model
    if "chair" in objects_to_load:
        opt.model3D_dir = join(model3D_dir, "chair_df")
        chair_model = TestModelSimple(opt)

    # load airplane model
    if "airplane" in objects_to_load:
        opt.model3D_dir = join(model3D_dir, "airplane_df")
        airplane_model = TestModelSimple(opt, flip=True)

    # load airplane model
    if "table" in objects_to_load:
        opt.model3D_dir = join(model3D_dir, "table_df")
        table_model = TestModelSimple(opt, flip=True)

    return car_model, chair_model, airplane_model, table_model
