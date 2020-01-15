from VON.models.base_model import BaseModel
import numpy as np
import torch
import trimesh
import scipy
import pdb

class TestModelSimple(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        assert (not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.vae = True
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['mask', 'depth', 'image']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and
        # base_model.load_networks
        self.model_names = ['G_AB', 'G_3D']
        self.model_names += 'E'
        self.cuda_names = ['z_shape', 'z_texture', 'rot_mat']
        self.use_df = opt.use_df or opt.dataset_mode.find('df') >= 0

        self.netG_3D = self.define_G_3D()
        #self.netG_AB = self.define_G(opt.input_nc, opt.output_nc, opt.nz_texture, ext='AB')
        #self.netE = self.define_E(opt.output_nc, self.vae)
        self.is_loaded = True
        self.nz_shape = opt.nz_shape
        self.nz_texture = opt.nz_texture
        self.setup_DR(opt)
        self.bg_B = 1
        self.bg_A = -1
        self.critCycle = torch.nn.L1Loss().to(self.device)

    def set_input(self, input, reset_shape=False, reset_texture=False):
        self.input_B = input[0]['image'].to(self.device)
        self.mask_B = input[0]['real_im_mask'].to(self.device)
        if reset_shape or not hasattr(self, 'voxel'):
            self.voxel = input[1]['voxel'].to(self.device)
        if reset_texture or not hasattr(self, 'z_texture'):
            with torch.no_grad():
                self.z_texture, mu, var = self.encode(self.input_B, vae=self.vae)
                self.z_texture = mu

    def load_obj(self, obj_path):
        # this follows the learning target creation used to train the 3D networks
        print("loading .obj file from {} ...".format(obj_path))
        res = 128
        density = 80000
        mesh = trimesh.load_mesh(obj_path)
        try:
            dump = mesh.dump()
            mesh = dump.sum()
        except:
            pass
        mesh.vertices -= mesh.centroid
        x = np.linspace(-0.5, 0.5, res)
        xv, yv, zv = np.meshgrid(x, x, x)
        points = np.zeros([res, res, res, 3])
        points[:, :, :, 0] = xv
        points[:, :, :, 1] = yv
        points[:, :, :, 2] = zv
        points = points.reshape(res**3, 3)
        mesh_points = mesh.sample(int(mesh.area * density))
        tree = scipy.spatial.cKDTree(mesh_points, leafsize=30)
        d, _ = tree.query(points, k=5)
        d = np.mean(d, 1)
        # distance function representation
        d_s = d.reshape(res, res, res)
        data_3d = torch.from_numpy(d_s).float().unsqueeze(0)
        data_3d = np.exp(-8.0 * data_3d)
        data_3d = data_3d.transpose(1, 2)
        data_3d = torch.flip(data_3d, [1])
        data_3d = data_3d.transpose(2, 3)
        data_3d = torch.flip(data_3d, [2])
        data_3d = data_3d.contiguous()
        data_3d = data_3d.unsqueeze(0).to(self.device)
        return data_3d

    def sample_3d(self, shape_code):
        assert shape_code.ndim == 1
        assert len(shape_code) == self.nz_shape
        with torch.no_grad():
            shape_code = torch.Tensor(shape_code).float().to(self.device)
            data_3d = self.netG_3D(shape_code.view(1, self.nz_shape, 1, 1, 1))
        return data_3d

    def render_3d(self, data_3d, view_code):
        assert view_code.ndim == 1
        assert len(view_code) == 2

        # voxel = self.sample_3d(shape_code.reshape((1, self.nz_shape, 1, 1, 1)))

        with torch.no_grad():
            rot_mat = self.azele2matrix(az=view_code[1], ele=view_code[0]).unsqueeze(0).repeat(1, 1, 1)
            rot_mat = rot_mat.to(self.device)
            mask, depth = self.get_depth(data_3d, rot_mat, use_df=self.use_df)

        return mask[0].cpu().numpy(), depth[0].cpu().numpy()

    @staticmethod
    def azele2matrix(az=0, ele=0):
        R0 = torch.zeros([3, 3])
        R = torch.zeros([3, 4])
        R0[0, 1] = 1
        R0[1, 0] = -1
        R0[2, 2] = 1
        az = az * np.pi / 180
        ele = ele * np.pi / 180
        cos = np.cos
        sin = np.sin
        R_ele = torch.FloatTensor(
            [[1, 0, 0], [0, cos(ele), -sin(ele)], [0, sin(ele), cos(ele)]])
        R_az = torch.FloatTensor(
            [[cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]])
        R_rot = torch.mm(R_az, R_ele)
        R_all = torch.mm(R_rot, R0)
        R[:3, :3] = R_all
        return R

    def update_D(self):
        pass

    def update_G(self):
        pass
