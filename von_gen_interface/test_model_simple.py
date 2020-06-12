from VON.models.base_model import BaseModel
import numpy as np
import torch
from skimage import measure

import pdb


class TestModelSimple(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt, flip=False):
        assert (not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.vae = True
        self.use_df = opt.use_df or opt.dataset_mode.find('df') >= 0
        self.iso_th = opt.iso_th
        self.voxel_res = opt.voxel_res

        self.netG_3D = self.define_G_3D()
        self.nz_shape = opt.nz_shape

        self.flip = flip

    def set_input(self, input, reset_shape=False, reset_texture=False):
        self.input_B = input[0]['image'].to(self.device)
        self.mask_B = input[0]['real_im_mask'].to(self.device)
        if reset_shape or not hasattr(self, 'voxel'):
            self.voxel = input[1]['voxel'].to(self.device)
        if reset_texture or not hasattr(self, 'z_texture'):
            with torch.no_grad():
                self.z_texture, mu, var = self.encode(self.input_B, vae=self.vae)
                self.z_texture = mu

    def sample_3d(self, shape_code, as_mesh=True):
        """Sample a 3D object """
        assert shape_code.ndim == 1
        assert len(shape_code) == self.nz_shape
        with torch.no_grad():
            shape_code = torch.Tensor(shape_code).float().to(self.device)
            data_3d = self.netG_3D(shape_code.view(1, self.nz_shape, 1, 1, 1))
        if not as_mesh:
            return data_3d
        else:
            # we want to have triangular meshes now instead of df representation
            data_3d = data_3d[0, 0, :, :, :]
            if self.flip:
                pass
                # data_3d = torch.flip(data_3d, [0])
                # data_3d = torch.flip(data_3d, [1])
                # data_3d = data_3d.transpose(1, 2)
                # data_3d = torch.flip(data_3d, [0])
                # data_3d = data_3d.transpose(0, 1)

            data_3d = data_3d.permute(1,2,0)

            space = 1.0 / float(self.voxel_res)

            verts, faces, _, _ = measure.marching_cubes_lewiner(data_3d.cpu().numpy(), 0.85, spacing=(space, space, space))

            verts -= 0.5
            verts = torch.from_numpy(verts).float().to("cuda")
            faces = np.ascontiguousarray(faces)
            faces = torch.from_numpy(faces).float().to("cuda").unsqueeze(0)
            return [verts, faces]

    def update_D(self):
        pass

    def update_G(self):
        pass
