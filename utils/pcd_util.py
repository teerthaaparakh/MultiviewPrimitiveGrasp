import numpy as np

def get_pcd(
    in_world=True,
    filter_depth=False,
    depth_min=0.20,
    depth_max=1.50,
    cam_ext_mat=None,
    rgb_image=None,
    depth_image=None,
    seg_image=None,
    cam_intr_mat=None,
):
    """
    Get the point cloud from the entire depth image
    in the camera frame or in the world frame.
    Returns:
        2-element tuple containing

        - np.ndarray: point coordinates (shape: :math:`[N, 3]`).
        - np.ndarray: rgb values (shape: :math:`[N, 3]`).
        - np.ndarray: seg values (shape: :math:`[N, 1]`).
    """

    rgb_im = rgb_image
    depth_im = depth_image
    seg_im = seg_image
    img_shape = rgb_im.shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    # pcd in camera from depth

    # assert (
    #     len(seg_im.shape) == 3
    # ), "the segmentation mask must be 3D with last dimension containing 1 channel"
    depth = depth_im.reshape(-1)

    rgb = None
    if rgb_im is not None:
        rgb = rgb_im.reshape(-1, 3)
    # if seg_im is not None:
    #     seg = seg_im.reshape(-1, 1)
    depth_min = depth_min
    depth_max = depth_max
    if filter_depth:
        valid = depth > depth_min
        valid = np.logical_and(valid, depth < depth_max)
        depth = depth[valid]
        if rgb is not None:
            rgb = rgb[valid]
        # if seg is not None:
        #     seg = seg[valid]
        uv_one_in_cam = get_uv_one_in_cam(cam_intr_mat, img_height, img_width)[:, valid]
    else:
        uv_one_in_cam = get_uv_one_in_cam(cam_intr_mat, img_height, img_width)
    pts_in_cam = np.multiply(uv_one_in_cam, depth)
    if not in_world:
        pcd_pts = pts_in_cam.T
        pcd_rgb = rgb
        # pcd_seg = seg
        return pcd_pts, pcd_rgb
    else:
        if cam_ext_mat is None:
            raise ValueError(
                "Please call set_cam_ext() first to set up"
                " the camera extrinsic matrix"
            )

        pts_in_cam = np.concatenate(
            (pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0
        )
        pts_in_world = np.dot(cam_ext_mat, pts_in_cam)
        pcd_pts = pts_in_world[:3, :].T
        pcd_rgb = rgb
        # pcd_seg = seg

        return pcd_pts, pcd_rgb


def get_combined_pcd(colors, depths, cams_extr, cam_intr, idx=None):
    pcd_pts = []
    pcd_rgb = []

    if idx is None:
        idx = list(range(len(colors)))

    count = 0
    for color, depth, cam_extr in zip(colors, depths, cams_extr):
        if count in idx:
            pts, rgb = get_pcd(
                cam_ext_mat=cam_extr,
                rgb_image=color,
                depth_image=depth,
                cam_intr_mat=cam_intr,
            )
            pcd_pts.append(pts)
            pcd_rgb.append(rgb)
        count += 1

    return np.concatenate(pcd_pts, axis=0), np.concatenate(pcd_rgb, axis=0)


def get_uv_one_in_cam(cam_intr_mat, img_height, img_width):
    cam_int_mat_inv = np.linalg.inv(cam_intr_mat)

    img_pixs = np.mgrid[0:img_height, 0:img_width].reshape(2, -1)
    img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
    _uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
    uv_one_in_cam = np.dot(cam_int_mat_inv, _uv_one)

    return uv_one_in_cam



def project_pcd_to_image(
    pcd, cam_intr, cam_extr, height, width, depth=False, background_val=0
):
    """
    pcd: Nx3
    cam_intr: 3x3
    c2w: rotation, position
    labels = if given, returns the projected labels, bounding box, and colormap
    """

    xyzs = np.asarray(pcd.points)
    colors = np.uint8(255 * np.asarray(pcd.colors))

    N = xyzs.shape[0]
    X_WC = cam_extr
    X_CW = np.linalg.inv(X_WC)
    pcd_cam = X_CW @ np.concatenate((xyzs, np.ones((N, 1))), axis=1).T
    pcd_cam = pcd_cam[:3, :].T

    iz = np.argsort(pcd_cam[:, -1])[::-1]
    pcd_cam, colors = pcd_cam[iz], colors[iz]

    fx = cam_intr[0, 0]
    fy = cam_intr[1, 1]
    cx = cam_intr[0, 2]
    cy = cam_intr[1, 2]

    px = (pcd_cam[:, 0] * fx / pcd_cam[:, 2]) + cx
    py = (pcd_cam[:, 1] * fy / pcd_cam[:, 2]) + cy

    px = np.clip(np.int32(px), 0, width - 1)
    py = np.clip(np.int32(py), 0, height - 1)

    colormap = background_val * np.ones((height, width, colors.shape[-1]))
    colormap[py, px, :] = colors

    if depth:
        # return colormap, None
        depthmap = np.zeros((height, width), dtype=np.float32)
        depthmap[py, px] = pcd_cam[:, 2]
        return colormap, depthmap

    return colormap, None