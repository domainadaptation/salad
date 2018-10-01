import numpy as np
import cv2
import torch

""" Taken from Self-Ensembling Github Repository
"""


def identity_xf(N):
    """
    Construct N identity 2x3 transformation matrices
    :return: array of shape (N, 2, 3)
    """
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    return xf


def inv_nx2x2(X):
    """
    Invert the N 2x2 transformation matrices stored in X; a (N,2,2) array
    :param X: transformation matrices to invert, (N,2,2) array
    :return: inverse of X
    """
    rdet = 1.0 / (X[:, 0, 0] * X[:, 1, 1] - X[:, 1, 0] * X[:, 0, 1])
    y = np.zeros_like(X)
    y[:, 0, 0] = X[:, 1, 1] * rdet
    y[:, 1, 1] = X[:, 0, 0] * rdet
    y[:, 0, 1] = -X[:, 0, 1] * rdet
    y[:, 1, 0] = -X[:, 1, 0] * rdet
    return y


def inv_nx2x3(m):
    """
    Invert the N 2x3 transformation matrices stored in X; a (N,2,3) array
    :param X: transformation matrices to invert, (N,2,3) array
    :return: inverse of X
    """
    m2 = m[:, :, :2]
    mx = m[:, :, 2:3]
    m2inv = inv_nx2x2(m2)
    mxinv = np.matmul(m2inv, -mx)
    return np.append(m2inv, mxinv, axis=2)


def cat_nx2x3(a, b):
    """
    Multiply the N 2x3 transformations stored in `a` with those in `b`
    :param a: transformation matrices, (N,2,3) array
    :param b: transformation matrices, (N,2,3) array
    :return: `a . b`
    """
    a2 = a[:, :, :2]
    b2 = b[:, :, :2]

    ax = a[:, :, 2:3]
    bx = b[:, :, 2:3]

    ab2 = np.matmul(a2, b2)
    abx = ax + np.matmul(a2, bx)
    return np.append(ab2, abx, axis=2)


def rotation_matrices(thetas):
    """
    Generate rotation matrices
    :param thetas: rotation angles in radians as a (N,) array
    :return: rotation matrices, (N,2,3) array
    """
    N = thetas.shape[0]
    rot_xf = np.zeros((N, 2, 3), dtype=np.float32)
    rot_xf[:, 0, 0] = rot_xf[:, 1, 1] = np.cos(thetas)
    rot_xf[:, 1, 0] = np.sin(thetas)
    rot_xf[:, 0, 1] = -np.sin(thetas)
    return rot_xf


def centre_xf(xf, size):
    """
    Centre the transformations in `xf` around (0,0), where the current centre is assumed to be at the
    centre of an image of shape `size`
    :param xf: transformation matrices, (N,2,3) array
    :param size: image size
    :return: centred transformation matrices, (N,2,3) array
    """
    height, width = size

    # centre_to_zero moves the centre of the image to (0,0)
    centre_to_zero = np.zeros((1, 2, 3), dtype=np.float32)
    centre_to_zero[0, 0, 0] = centre_to_zero[0, 1, 1] = 1.0
    centre_to_zero[0, 0, 2] = -float(width) * 0.5
    centre_to_zero[0, 1, 2] = -float(height) * 0.5

    # centre_to_zero then xf
    xf_centred = cat_nx2x3(xf, centre_to_zero)

    # move (0,0) back to the centre
    xf_centred[:, 0, 2] += float(width) * 0.5
    xf_centred[:, 1, 2] += float(height) * 0.5

    return xf_centred


class ImageAugmentation (object):
    def __init__(self, hflip, xlat_range, affine_std, rot_std=0.0,
                 intens_flip=False,
                 intens_scale_range_lower=None, intens_scale_range_upper=None,
                 intens_offset_range_lower=None, intens_offset_range_upper=None,
                 scale_x_range=None, scale_y_range=None, scale_u_range=None, gaussian_noise_std=0.0,
                 blur_range=None):
        self.hflip = hflip
        self.xlat_range = xlat_range
        self.affine_std = affine_std
        self.rot_std = rot_std
        self.intens_scale_range_lower = intens_scale_range_lower
        self.intens_scale_range_upper = intens_scale_range_upper
        self.intens_offset_range_lower = intens_offset_range_lower
        self.intens_offset_range_upper = intens_offset_range_upper
        self.intens_flip = intens_flip
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range
        self.scale_u_range = scale_u_range
        self.gaussian_noise_std = gaussian_noise_std
        self.blur_range = blur_range

    def augment(self, X):
        X = X.copy()
        xf = identity_xf(len(X))

        if self.hflip:
            x_hflip = np.random.binomial(1, 0.5, size=(len(X),)) * 2 - 1
            xf[:, 0, 0] = x_hflip

        if self.scale_x_range is not None and self.scale_x_range[0] is not None:
            xf[:, 0, 0] *= np.random.uniform(low=self.scale_x_range[0],
                                             high=self.scale_x_range[1], size=(len(X),))
        if self.scale_y_range is not None and self.scale_y_range[0] is not None:
            xf[:, 1, 1] *= np.random.uniform(low=self.scale_y_range[0],
                                             high=self.scale_y_range[1], size=(len(X),))
        if self.scale_u_range is not None and self.scale_u_range[0] is not None:
            scale_u = np.random.uniform(
                low=self.scale_u_range[0], high=self.scale_u_range[1], size=(len(X),))
            xf[:, 0, 0] *= scale_u
            xf[:, 1, 1] *= scale_u

        if self.affine_std > 0.0:
            xf[:, :, :2] += np.random.normal(scale=self.affine_std, size=(len(X), 2, 2))

        if self.rot_std > 0.0:
            thetas = np.random.normal(scale=self.rot_std, size=(len(X),))
            rot_xf = rotation_matrices(thetas)
            xf = cat_nx2x3(xf, rot_xf)

        if self.xlat_range > 0.0:
            xf[:, :, 2:] += np.random.uniform(low=-self.xlat_range,
                                              high=self.xlat_range, size=(len(X), 2, 1))

        if self.intens_flip:
            col_factor = (np.random.binomial(1, 0.5, size=(
                len(X), 1, 1, 1)) * 2 - 1).astype(np.float32)
            X = (X * col_factor).astype(np.float32)

        if self.intens_scale_range_lower is not None:
            col_factor = np.random.uniform(low=self.intens_scale_range_lower, high=self.intens_scale_range_upper,
                                           size=(len(X), 1, 1, 1))

            X = (X * col_factor).astype(np.float32)

        if self.intens_offset_range_lower is not None:
            col_offset = np.random.uniform(low=self.intens_offset_range_lower, high=self.intens_offset_range_upper,
                                           size=(len(X), 1, 1, 1))

            X = (X + col_offset).astype(np.float32)

        xf_centred = centre_xf(xf, X.shape[2:])
        for i in range(len(X)):
            if X.shape[1] == 1:
                X[i, 0, :, :] = cv2.warpAffine(
                    X[i, 0, :, :], xf_centred[i, :, :], (X.shape[3], X.shape[2]))
            else:
                X[i, :, :, :] = cv2.warpAffine(X[i, :, :, :].transpose(
                    1, 2, 0), xf_centred[i, :, :], (X.shape[3], X.shape[2])).transpose(2, 0, 1)

        if self.blur_range is not None and self.blur_range[0] is not None:
            sigmas = np.random.uniform(
                low=self.blur_range[0], high=self.blur_range[1], size=(len(X),))
            sigmas = np.maximum(sigmas, 0.0)
            for i in range(len(X)):
                sigma = sigmas[i]
                # ksize must be odd number
                ksize = int(sigma+0.5) * 8 + 1
                if X.shape[1] == 1:
                    X[i, 0, :, :] = cv2.GaussianBlur(X[i, 0, :, :], (ksize, ksize), sigmaX=sigma)
                else:
                    X[i, :, :, :] = cv2.GaussianBlur(X[i, :, :, :].transpose(
                        1, 2, 0), (ksize, ksize), sigmaX=sigma).transpose(2, 0, 1)

        if self.gaussian_noise_std > 0.0:
            X += np.random.normal(scale=self.gaussian_noise_std, size=X.shape).astype(np.float32)

        return X

    def augment_pair(self, X):
        return self.augment(X), self.augment(X)


class Augmentation():

    def __init__(self, dataset, n_samples=1):
        self.transformer = ImageAugmentation(
            affine_std=0.1,
            gaussian_noise_std=0.1,
            hflip=False,
            intens_flip=True,
            intens_offset_range_lower=-.5, intens_offset_range_upper=.5,
            intens_scale_range_lower=0.25, intens_scale_range_upper=1.5,
            xlat_range=2.0
        )

        self.dataset = dataset
        self.n_samples = n_samples

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        x, y = self.dataset[index]

        X = torch.stack([x.clone() for _ in range(self.n_samples)], dim=0)
        X = self.transformer.augment(X.numpy())

        outp = [torch.from_numpy(x).float() for x in X] + [y, ]

        return outp
