from os.path import join
from VON.von_gen_interface.test_model_simple import TestModelSimple
from VON.von_gen_interface.options import opt
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


def load_models(gpu_ids, car=True, chair=True):
    # preprocess opt
    preprocess(opt, gpu_ids)

    car_model = None
    chair_model = None

    model3D_dir = opt.model3D_dir
    model2D_dir = opt.model2D_dir

    # load car_model
    if car:
        opt.model3D_dir = join(model3D_dir, "car_df")
        opt.model2D_dir = join(model2D_dir, "car_df/latest")
        car_model = TestModelSimple(opt)

    # load chair model
    if chair:
        opt.model3D_dir = join(model3D_dir, "chair_df")
        opt.model2D_dir = join(model2D_dir, "chair_df/latest")
        chair_model = TestModelSimple(opt)

    return car_model, chair_model
