from os.path import join, dirname
from munch import Munch

# Define opt dictionary
opt = Munch()

# basic options
opt.root = join(dirname(__file__), "../")
opt.dataroot = None
opt.batch_size = 1
opt.load_size = 128
opt.crop_size = 128
opt.input_nc = 1
opt.output_nc = 3
opt.nz_texture = 8
opt.nz_shape = 200
# opt.gpu_ids = None
opt.name = "test_mode"
opt.model = "shape_gan"
opt.epoch = "latest"
opt.phase = "val"
opt.num_threads = 1
opt.checkpoints_dir = join(opt.root, "results_texture")
opt.display_winsize = 128

# dataset
opt.dataset_mode = "image_and_df"
opt.resize_or_crop = "crop_real_im"
opt.serial_batches = False
opt.max_dataset_size = float("inf")

# models
opt.num_Ds = 2
opt.netD = "multi"
opt.netG = "resnet_cat"
opt.use_dropout = False
opt.netE = "adaIN"
opt.where_add = "all"
opt.netG_3D = "G0"
opt.netD_3D = "D0"
opt.norm = "inst"
opt.nl = "relu"
opt.G_norm_3D = "batch3d"
opt.D_norm_3D  ="none"

# number of channels in networks
opt.nef = 64
opt.ngf = 64
opt.ndf = 64
opt.ngf_3d = 64
opt.ndf_3d = 64

# extra parameters
opt.gan_mode = "lsgan"
opt.init_type = "kaiming"
opt.init_param = 0.02

# 3D parameters
opt.voxel_res = 128
# opt.class_3d
opt.model3D_dir = join(opt.root, "final_models/models_3D")
opt.model2D_dir = join(opt.root, "final_models/models_2D")
opt.use_df = False
opt.df_th = 0.9

# misc
opt.no_largest = False
opt.crop_align = False
opt.suffix = ""
opt.verbose = False
opt.print_grad = False
opt.seed = 0
opt.isTrain = False
opt.aspect_ratio = 1.0

# relevant for dataset
opt.random_shift = False
opt.color_jitter = False
opt.pose_type = "hack"
opt.pose_align = False
opt.no_flip = False

# relevant for  ShapeGANModel
opt.lambda_GAN_3D = 1.
opt.lambda_gp_3D = 10
opt.gan_mode_3D = "wgangp"
opt.gp_norm_3D = 1.
opt.gp_type_3D = "mixed"
opt.vis_batch_num = 2
opt.lr_3d = 0.0001
