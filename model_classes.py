import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 
import math 

class ShiftModule(nn.Module):
    def __init__(self, height, width, learn_scale=False):
        super(ShiftModule, self).__init__()
        self.x_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.y_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        grid = torch.stack((x_coords, y_coords), dim=-1).float()
        grid[:, :, 0] = 2.0 * grid[:, :, 0] / (width - 1) - 1.0
        grid[:, :, 1] = 2.0 * grid[:, :, 1] / (height - 1) - 1.0
        self.grid = nn.Parameter(grid[None], requires_grad=False)
        self.rotation = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1), requires_grad=learn_scale)

    def forward(self, x):
        grid = self.grid.clone().to(x.dtype) * self.scale

        cos_theta = torch.cos(self.rotation)
        sin_theta = torch.sin(self.rotation)
        rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]]).to(x.device).to(x.dtype)
        grid = torch.matmul(grid.view(-1, 2), rotation_matrix).view(1, x.shape[-2], x.shape[-1], 2)

        grid[..., 0] = grid[..., 0] + self.x_shift
        grid[..., 1] = grid[..., 1] + self.y_shift

        out = F.grid_sample(x[None], grid, mode='bilinear', padding_mode='reflection', align_corners=False)

        return out[0]
    

class GridOffsetModule(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        grid = torch.zeros((height, width))
        self.grid = nn.Parameter(grid[None], requires_grad=True)

    def get_res(self):
        res = self.grid
        return res

    def forward(self, x=None):
        res = self.grid
        if x is None:
            return res
        out = res + x
        return out
    

class FlatFieldingModule(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        grid = torch.ones((height, width))
        self.grid = nn.Parameter(grid[None], requires_grad=True)

    def get_res(self):
        res = self.grid
        return res

    def forward(self, x=None):
        res = self.grid
        if x is None:
            return res
        out = res * x
        return out
    
class PerMirrorZernikes(nn.Module):
    """
    Parametrizes the OPD at the entrance of JWST as N Zernike polynomials for each one of the hexagonal mirror segments
    """
    def __init__(self,labelled_transmission,segment_centers, segment_labels,npixels, num_zernikes=3,DEVICE='cuda'):
        super().__init__()

        self.zernike_coeffs_per_mirror = nn.Parameter(torch.zeros(num_zernikes,18, device=DEVICE), requires_grad=True) # 18 mirror segments
        self.num_zernikes = num_zernikes
        # Origin 0,0 at the corner, to match the origin of the segment_centers keyword.
        # Only used to shift the origin of the Zernikes to the center of each segment
        coords = np.meshgrid(np.linspace(0.,npixels, npixels), np.linspace(0.,npixels, npixels))
        self.coords_grid_x = torch.from_numpy(coords[0]).to(DEVICE)[None]
        self.coords_grid_y = torch.from_numpy(coords[1]).to(DEVICE)[None]
        self.segment_radius = nn.Parameter((npixels / 5. / 2.) / torch.cos(torch.deg2rad(torch.tensor(30.))), requires_grad=False) # in pixels
        self.device = DEVICE
        ordered_segment_masks = []
        ordered_segment_centers = []

        for seg_name, seg_label in segment_labels.items():
            current_mirror_transmission = np.where(labelled_transmission == seg_label, 1.,0.)
            # Flip to match orientation of our simulator
            ordered_segment_masks.append(torch.from_numpy(np.flip(current_mirror_transmission, axis=0).copy())[None])

            # Invert y-axis of center coordinates, to match orientation of our simulator
            center_x = segment_centers[seg_name][0]
            center_y =npixels - segment_centers[seg_name][1]
            ordered_segment_centers.append([center_x, center_y])
        
        self.ordered_segment_masks = torch.stack(ordered_segment_masks).to(DEVICE)
        self.ordered_segment_centers = torch.tensor(ordered_segment_centers, dtype=torch.float64).to(DEVICE)

        per_mirror_zernike_basis = torch.zeros(num_zernikes,1024,1024).to(DEVICE) # this should have shape (N_zernikes, 1024, 1024) # bruhhh

        for i in range(len(self.ordered_segment_masks)):
            r_grid = torch.sqrt((self.coords_grid_x - self.ordered_segment_centers[i][0])**2 + (self.coords_grid_y-self.ordered_segment_centers[i][1])**2)
            theta_grid = torch.arctan2(self.coords_grid_y-self.ordered_segment_centers[i][1], self.coords_grid_x- self.ordered_segment_centers[i][0])
            curr_mirr = self._generate_zernike_basis_normalized(r_grid, theta_grid,max_num_zern=self.num_zernikes)

            per_mirror_zernike_basis+=curr_mirr[:,0,:,:] * self.ordered_segment_masks[i, 0].expand_as(curr_mirr[:,0,:,:]) 
            
        self.zernike_basis_per_mirror = per_mirror_zernike_basis


    def _zernike_radial(self, n, m, r):
        R = torch.zeros_like(r)
        for k in range((n - abs(m)) // 2 + 1):
            coeff = ((-1)**k * math.comb(n - k, k) * math.comb(n - 2*k, (n - abs(m)) // 2 - k))
            R += coeff * r**(n - 2*k)
        return R

    def _zernike_polynomial(self, n, m, radius_grid, theta_grid):
        R = self._zernike_radial(n, m, radius_grid)
        if m > 0:
            return R * torch.cos(m * theta_grid)
        elif m < 0:
            return R * torch.sin(-m * theta_grid)
        else:
            return R    

    def _generate_basis(self,radius_grid, theta_grid,max_num_zern=5):
        basis = []
        count = 0
        n = 0
        while count < max_num_zern:
            for m in range(-n, n + 1, 2):
                basis.append(self._zernike_polynomial(n, m,radius_grid,theta_grid)/self.segment_radius) # Double check this
                count += 1
                if count >= max_num_zern:
                    break
            n += 1
        return torch.stack(basis, dim=0).to(self.device)  # Shape: (N, H, W)
    

    def _generate_zernike_basis_normalized(self, radius_array, theta_array,max_num_zern=5):
        """
        Generate all Zernike polynomials up to degree N (inclusive).
        
        Args:
            N: Maximum radial degree
            radius_array: 2D tensor of radii in pixels
            theta_array: 2D tensor of angles in radians
            R: Aperture radius in pixels

        Returns:
            basis: Tensor of shape (num_modes, H, W), RMS-normalized within aperture
            nm_list: List of (n, m) tuples corresponding to each mode
        """
        rho = radius_array / self.segment_radius
        mask = (rho <= 1.0).float()
        
        basis_list = []
        count = 0
        n = 0
        while count < max_num_zern:
            for m in range(-n, n + 1, 2):  # m steps by 2 to ensure (n - |m|) even
                if (n - abs(m)) % 2 != 0:
                    continue  # skip invalid combinations
                Z = self._zernike_polynomial(n, m, rho, theta_array)
                Z = Z * mask  # zero out values outside the unit circle

                # Normalize to RMS = 1 inside the aperture
                mean_sq = torch.sum(Z**2 * mask) / torch.sum(mask)
                Z_norm = Z / torch.sqrt(mean_sq + 1e-12)

                basis_list.append(Z_norm)
                count += 1

                if count == max_num_zern:
                    break
            n += 1
        basis_tensor = torch.stack(basis_list, dim=0)  # shape: (num_modes, H, W)
        return basis_tensor.to(self.device)


    def get_res(self):
        wf = torch.zeros(1,1024,1024).to(self.device)
        for nthzern in range(self.num_zernikes):
            # if nthzern == 4: # only defocus to test
            for nthmirror in range(len(self.ordered_segment_masks)):
                wf[0]+= self.zernike_basis_per_mirror[nthzern] * self.zernike_coeffs_per_mirror[nthzern,nthmirror] * self.ordered_segment_masks[nthmirror,0] * 1e-1

        return wf 
    
    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out
    

class OPDOffsetModule(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        grid = torch.zeros((1, height, width))
        self.grid = nn.Parameter(grid, requires_grad=True)

    def get_res(self):
        res = self.grid
        return res

    def total_variation_loss(self):
        # Scale the grid by its mean
        scaled_grid = self.grid / (torch.mean(torch.abs(self.grid)) + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Calculate differences in x and y directions
        diff_x = torch.abs(scaled_grid[:, :, 1:] - scaled_grid[:, :, :-1])
        diff_y = torch.abs(scaled_grid[:, 1:, :] - scaled_grid[:, :-1, :])
        
        # Sum up the differences
        tv_loss = torch.sum(diff_x) + torch.sum(diff_y)
        
        return tv_loss

    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out
    
class FluxOffsetModule(nn.Module):
    def __init__(self, flux=None):
        super().__init__()
        # self.data = nn.Parameter(torch.ones(1), requires_grad=True)
        if flux is None:
            self.data = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.data = nn.Parameter(flux, requires_grad=True)

    def forward(self, x):
        return self.data * x
    
class AngleOffsetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = nn.Parameter(torch.zeros(2), requires_grad=True)

    def forward(self):

        return self.data
    
class LinearInterpOPD(nn.Module):
    """
    Linear interpolation between two OPD measurements
    """
    def __init__(self, before_opd, after_opd, coeff_init=0.):
        super().__init__()
        self.coeff = nn.Parameter(torch.FloatTensor([coeff_init]), requires_grad=True)
        self.before_opd = before_opd
        self.after_opd = after_opd

    def get_res(self):
        return (self.after_opd - self.before_opd) * self.coeff
    
    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out
    
class PTT_OPD(nn.Module):
    """
    Parametrizes the OPD at the entrance of JWST as Piston-Tip-Tilt of the hexagonal mirror segments
    """
    def __init__(self,labelled_transmission,segment_centers, segment_labels,npixels, DEVICE='cuda'):
        super().__init__()
        self.piston_coeffs = nn.Parameter(torch.zeros(18))
        self.xtilt_coeffs = nn.Parameter(torch.zeros(18))
        self.ytilt_coeffs = nn.Parameter(torch.zeros(18))
        # Origin 0,0 at the corner, to match the origin of the segment_centers keyword.
        # Only used to shift the origin of the Zernikes to the center of each segment
        coords = np.meshgrid(np.linspace(0.,npixels, npixels), np.linspace(0.,npixels, npixels))
        self.coords_grid_x = torch.from_numpy(coords[0]).to(DEVICE)[None]
        self.coords_grid_y = torch.from_numpy(coords[1]).to(DEVICE)[None]
        self.segment_radius = nn.Parameter((npixels / 5. / 2.) / torch.cos(torch.deg2rad(torch.tensor(30.))), requires_grad=False) # in pixels

        ordered_segment_masks = []
        ordered_segment_centers = []

        for seg_name, seg_label in segment_labels.items():
            current_mirror_transmission = np.where(labelled_transmission == seg_label, 1.,0.)
            # Flip to match orientation of our simulator
            ordered_segment_masks.append(torch.from_numpy(np.flip(current_mirror_transmission, axis=0).copy())[None])

            # Invert y-axis of center coordinates, to match orientation of our simulator
            center_x = segment_centers[seg_name][0]
            center_y =npixels - segment_centers[seg_name][1]
            ordered_segment_centers.append([center_x, center_y])
        
        self.ordered_segment_masks = torch.stack(ordered_segment_masks).to(DEVICE)
        self.ordered_segment_centers = torch.tensor(ordered_segment_centers, dtype=torch.float64).to(DEVICE)

    def get_res(self):
        total_opd = torch.zeros_like(self.coords_grid_x)
        for i in range(len(self.ordered_segment_masks)):
            piston_opd = torch.zeros_like(self.ordered_segment_masks[i])+ self.piston_coeffs[i] # simply add constant 
            r_grid = torch.sqrt((self.coords_grid_x - self.ordered_segment_centers[i][0])**2 + (self.coords_grid_y-self.ordered_segment_centers[i][1])**2)
            theta_grid = torch.arctan2(self.coords_grid_y-self.ordered_segment_centers[i][1], self.coords_grid_x- self.ordered_segment_centers[i][0])

            xtilt_opd = self.xtilt_coeffs[i] * 2* r_grid * torch.cos(theta_grid) / self.segment_radius # Normalize to unit radius
            ytilt_opd = self.ytilt_coeffs[i] * 2* r_grid * torch.sin(theta_grid) / self.segment_radius

            total_opd+=(piston_opd + xtilt_opd + ytilt_opd)* self.ordered_segment_masks[i]

        return total_opd
    
    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out
    
class LearnableGaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, init_sigma=0.28):
        super().__init__()
        self.kernel_size = kernel_size
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma)))  # Learn in log-space

    def forward(self, x):
        # x: (1, 1, N, H, W)
        if x.dim() != 5 or x.size(0) != 1 or x.size(1) != 1:
            raise ValueError("Input must have shape (1, 1, N, H, W)")

        sigma = torch.exp(self.log_sigma)
        coords = torch.arange(self.kernel_size, dtype=torch.double, device=x.device) - self.kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma)**2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, K, K)

        B, C, N, H, W = x.shape
        x_reshaped = x.view(B * N, C, H, W)  # Merge batch and depth

        blurred = F.conv2d(x_reshaped, kernel_2d, padding='same')

        blurred = blurred.view(B, C, N, H, W)  # Restore original shape

        return blurred
    

## ZERNIKE POLYNOMIALS PARAMETRIZATION
class ZernikeBasis:
    def __init__(self, height, width, max_modes):
        self.H = height
        self.W = width
        self.N = max_modes
        self.r, self.theta = self._create_polar_grid()
        self.mask = self.r <= 1.0
        self.basis = self._generate_basis()

    def _create_polar_grid(self):
        y, x = torch.meshgrid(torch.linspace(-1, 1, self.H),
                              torch.linspace(-1, 1, self.W),
                              indexing='ij')
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        r[r > 1] = 0  # Mask outside the unit disk
        return r, theta

    def _zernike_radial(self, n, m, r):
        R = torch.zeros_like(r)
        for k in range((n - abs(m)) // 2 + 1):
            coeff = ((-1)**k * math.comb(n - k, k) * math.comb(n - 2*k, (n - abs(m)) // 2 - k))
            R += coeff * r**(n - 2*k)
        return R

    def _zernike_polynomial(self, n, m):
        R = self._zernike_radial(n, m, self.r)
        if m > 0:
            return R * torch.cos(m * self.theta)
        elif m < 0:
            return R * torch.sin(-m * self.theta)
        else:
            return R

    def _generate_basis(self):
        basis = []
        count = 0
        n = 0
        while count < self.N:
            for m in range(-n, n + 1, 2):
                basis.append(self._zernike_polynomial(n, m))
                count += 1
                if count >= self.N:
                    break
            n += 1
        return torch.stack(basis, dim=0)  # Shape: (N, H, W)

    def get_basis(self):
        return self.basis.clone()

    def get_mask(self):
        return self.mask.clone()

class ZernikeModel(nn.Module):
    def __init__(self, zernike_basis: ZernikeBasis, DEVICE='cuda'):
        super().__init__()
        self.basis = zernike_basis.get_basis().to(DEVICE)[None]  # (1,N, H, W)
        self.mask = zernike_basis.get_mask().to(DEVICE)[None]
        self.coeffs = nn.Parameter(torch.zeros(1, self.basis.shape[1], device=DEVICE), requires_grad=True) # Learnable coefficients

    def get_res(self):
        wf = torch.sum(self.coeffs[:,:,None,None] * self.basis, dim=1)* self.mask 
        # print('---- req grad???', self.coeffs.requires_grad)
        return wf  # Mask outside unit disk

    def total_variation_loss(self):
        # Scale the grid by its mean
        scaled_grid = self.grid / (torch.mean(torch.abs(self.grid)) + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Calculate differences in x and y directions
        diff_x = torch.abs(scaled_grid[:, :, 1:] - scaled_grid[:, :, :-1])
        diff_y = torch.abs(scaled_grid[:, 1:, :] - scaled_grid[:, :-1, :])
        
        # Sum up the differences
        tv_loss = torch.sum(diff_x) + torch.sum(diff_y)
        
        return tv_loss

    def forward(self, x):
        res = self.get_res()
        out = res + x
        return out