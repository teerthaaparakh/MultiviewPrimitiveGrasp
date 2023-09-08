def get_kpts_3d(pose, width, cam_extr, world=False):
    """
    pose: 4x4 single grasp pose
    width: float
    Returns
        kpts_3d: 3x4 (xyz for each of the four keypoints)
    """
    # width = CANONICAL_LEN
    length = STICK_LEN * 3 / 4.0
    kpts_local_vertex = [
        [0, 0, width / 2],
        [-length, 0, width / 2],
        [-length, 0, -width / 2],
        [0, 0, -width / 2],
    ]
    kpts_3d = pose @ np.concatenate((kpts_local_vertex, np.ones((4, 1))), axis=1).T
    if world:
        return kpts_3d[:3, :].T
    else:
        X_WC = cam_extr
        X_CW = np.linalg.inv(X_WC)
        kpts_3d_cam = X_CW @ kpts_3d
        return kpts_3d_cam[:3, :].T


def get_kpts_2d(kpts_3d_cam, cam_intr):
    """
    kpts_3d: 4x3
    cam_extr: 4x4
    cam_intr: 3x3

    Returns
        4x2 (x, y projection of the 4 keypoints on the image)
    """
    cam_intr = np.array(cam_intr)
    fx = cam_intr[0, 0]
    fy = cam_intr[1, 1]
    cx = cam_intr[0, 2]
    cy = cam_intr[1, 2]

    px = (kpts_3d_cam[:, 0] * fx / kpts_3d_cam[:, 2]) + cx
    py = (kpts_3d_cam[:, 1] * fy / kpts_3d_cam[:, 2]) + cy
    return np.stack((px, py), axis=-1)


def get_kpts_2d_validity(kpts_2d, img_height, img_width):
    """
    kpts_2d: 4x2
    Returns:
        valid: scalar
    """
    px, py = kpts_2d[:, 0], kpts_2d[:, 1]
    if (px < 0).all() or (px >= img_width).all():
        # logging.warn("Projected keypoint is outside the image [x].")
        return False
    elif (py < 0).all() or (py >= img_height).all():
        logging.warn("Projected keypoint is outside the image [y].")
        return False
    return True


def kpts_to_offset(kpts_2d_raw, center, scale, width, height, t="np"):
    """
    kpts_2d_raw: 4x2
    center: 2
    scale: float
    width: int
    hieght: int
    """
    if t == "np":
        offsets = scale * (kpts_2d_raw - center) / np.array([width, height])
    if t == "torch":
        offsets = scale * (kpts_2d_raw - center) / torch.tensor([width, height])
    return offsets


def offset_to_kpts(offsets, center, scale, width, height, t="np"):
    if t == "np":
        kpts_2d_raw = offsets * np.array([width, height]) / scale + center
    if t == "torch":
        kpts_2d_raw = offsets * torch.tensor([width, height]) / scale + center
    return kpts_2d_raw


def get_kpts_2d_detectron(
    kpts_2d: np.ndarray, kpts_3d_cam: np.ndarray, depth: np.ndarray
):
    """
    obtains scale, center and keypoints offset (with visibility vals) for detectron

    kpts_2d: 4x2
    kpts_3d_cam: 4x3
    img_width, img_height: int, int
    depth: array of shape (img_height, img_width)

    Returns: dictionary
        offset: 4x3 (x, y, v) for each of the four points (normalized and scaled by
                                                           center depth)
        scale: float
        center: 3  (x, y, v) for the center
    """

    assert kpts_2d.shape == (4, 2)
    assert kpts_3d_cam.shape == (4, 3)

    h, w = depth.shape

    valid = get_kpts_2d_validity(kpts_2d, h, w)
    v = np.ones(4)
    center_2d = (kpts_2d[0] + kpts_2d[3]) / 2

    if valid:
        center_3d_cam = (kpts_3d_cam[0] + kpts_3d_cam[3]) / 2
        scale = center_3d_cam[2]

        px, py = kpts_2d[:, 0], kpts_2d[:, 1]
        px = np.clip(np.int32(px), 0, w - 1)
        py = np.clip(np.int32(py), 0, h - 1)

        clipped_kpts_2d = np.stack((px, py), axis=-1)
        # offsets = scale * (clipped_kpts_2d - center_2d) / np.array([w, h])
        offsets = scale * (clipped_kpts_2d - center_2d) / NORMALIZATION_CONST

        assert offsets.shape == (4, 2)

        # offsets = clipped_kpts_2d

        depth_val = depth[clipped_kpts_2d[:, 1], clipped_kpts_2d[:, 0]]
        kpts_depth = kpts_3d_cam[:, 2]
        v[depth_val > kpts_depth] = 2

    else:
        # putting in dummy values, matching the dimensions
        offsets = np.zeros_like(kpts_2d)
        scale = 1.0

    return {
        "offset_kpts": np.concatenate((offsets, v.reshape((4, 1))), axis=1),
        "center_2d": np.array([*(center_2d), 2]),
        "scale": scale,
        "valid": valid,
    }
