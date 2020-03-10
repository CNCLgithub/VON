from VON.models.base_model import BaseModel
import numpy as np
import torch
import trimesh
import scipy
from skimage import measure
import kaolin as kal
import neural_renderer as nr
import mahotas as mht
import pdb

class TestModelSimple(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt, use_kaolin=True):
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
        self.iso_th = opt.iso_th
        self.voxel_res = opt.voxel_res

        self.netG_3D = self.define_G_3D()
        #self.netG_AB = self.define_G(opt.input_nc, opt.output_nc, opt.nz_texture, ext='AB')
        #self.netE = self.define_E(opt.output_nc, self.vae)
        self.is_loaded = True
        self.nz_shape = opt.nz_shape
        self.nz_texture = opt.nz_texture
        self.bg_B = 1
        self.bg_A = -1
        self.critCycle = torch.nn.L1Loss().to(self.device)
        self.use_kaolin = use_kaolin
        self.camera_distance = 1.0
        light_dir = np.array([0.1, 0, -1])
        light_dir = light_dir / np.linalg.norm(light_dir)
        light_intensity_directional = 0.9
        light_intensity_ambient = 0
        res = 224
        self.renderer = None
        if not self.use_kaolin:
            self.setup_DR(opt)
        else:
            self.renderer = nr.Renderer(camera_mode="look_at", image_size=res, light_direction=light_dir,
                                        light_intensity_directional=light_intensity_directional,
                                        light_intensity_ambient=light_intensity_ambient)

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

    def render_3d(self, data_3d, view_code, img_type="shading", df=True):
        assert view_code.ndim == 1
        assert len(view_code) == 3

        if self.use_kaolin:
            if df:
                with torch.no_grad():
                    # verts, faces = kal.conversion.sdf_to_trianglemesh(data_3d)
                    space = 1.0 / float(self.voxel_res)
                    data_3d = data_3d.cpu().numpy()[0, 0]
                    data_3d = -np.log(data_3d) / 8.
                    verts, faces, _, _ = measure.marching_cubes_lewiner(data_3d, self.iso_th,
                                                                        spacing=(space, space, space))
                    verts, faces, textures = self.prepare_mesh(verts, faces)
                    # rotate mesh such that it matches canonical view later
                    rot_matrix = torch.from_numpy(self.rot_matrix_from_deg(-90, -90, 0)).cuda()
                    verts = torch.matmul(rot_matrix, verts.squeeze().t()).t().cuda()
                    verts = verts[None, :, :].cuda()

                    image = self.render_image(verts, faces, textures, self.renderer, img_type=img_type,
                                        camera_distance=self.camera_distance,
                                        elevation=view_code[0], azimuth=view_code[1], tilt=view_code[2])
            else:
                print("Not yet implemented.")

        else:
            with torch.no_grad():
                rot_mat = self.azele2matrix(az=view_code[1], ele=view_code[0]).unsqueeze(0).repeat(1, 1, 1)
                rot_mat = rot_mat.to(self.device)
                mask, depth = self.get_depth(data_3d, rot_mat, use_df=self.use_df)
                if img_type == "silhouette":
                    image = mask[0].cpu().numpy()
                elif img_type == "depth":
                    image = depth[0].cpu().numpy()

        return image

    def render_image(self, vertices, faces, textures, renderer, img_type, camera_distance=2, elevation=0, azimuth=0, tilt=0):
        # rotate mesh according to Euler angles
        rot_matrix = torch.from_numpy(self.rot_matrix_from_deg(elevation, azimuth, tilt)).cuda()
        rot_vertices = torch.matmul(rot_matrix, vertices.squeeze().t()).t()
        rot_vertices = rot_vertices[None, :, :].cuda()
        renderer.eye =  nr.get_points_from_angles(camera_distance, 0, 0)
        image = None

        if img_type == "silhouette":
            # render silhouettes
            image = renderer(rot_vertices, faces, textures, mode="silhouettes").detach().cpu().numpy()[0]

        elif img_type == "shading" or img_type == "mooney":
            # render shading
            image = renderer(rot_vertices, faces, textures, mode="rgb").detach().cpu().numpy()[0].transpose((1, 2, 0))
            image = np.mean(image, axis=2)
            if img_type == "mooney":
                # make mooney by thresholding shading
                image = image * 255
                image = mht.median_filter(image, np.ones((5,) * len(image.shape), image.dtype))
                mooney_th = mht.otsu(image.astype(np.uint8), ignore_zeros=True)
                image = image > mooney_th
        else:
            print("Image type not recognized. Must be 'silhouette', 'shading' or 'mooney'.")

        return image

    @staticmethod
    def prepare_mesh(vertices, faces):
        vertices = torch.from_numpy(vertices).float()
        faces = torch.from_numpy(np.flip(faces, axis=0).copy()).int()
        # faces = faces.int()
        face_textures = (faces).clone()

        vertices = vertices[None, :, :].cuda()
        faces = faces[None, :, :].cuda()
        face_textures[None, :, :].cuda()

        # normalize verts
        vertices_max = vertices.max()
        vertices_min = vertices.min()
        vertices_middle = (vertices_max + vertices_min)/2.
        vertices = vertices - vertices_middle

        coef = 1
        vertices = vertices * coef

        textures = torch.ones(1, faces.shape[1], 1, 1, 1, 3, dtype=torch.float32).cuda()

        return vertices, faces, textures

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

    @staticmethod
    def rot_matrix_from_deg(u, v, w):
        u, v, w = np.deg2rad(u), np.deg2rad(v), np.deg2rad(w)
        rot_x = np.array([[1, 0, 0], [0, np.cos(u), -np.sin(u)], [0, np.sin(u), np.cos(u)]], dtype=np.float32)
        rot_y = np.array([[np.cos(v), 0, np.sin(v)], [0, 1, 0], [-np.sin(v), 0, np.cos(v)]], dtype=np.float32)
        rot_z = np.array([[np.cos(w), -np.sin(w), 0], [np.sin(w), np.cos(w), 0], [0, 0, 1]], dtype=np.float32)
        return rot_z.dot(rot_y.dot(rot_x))

    def update_D(self):
        pass

    def update_G(self):
        pass
