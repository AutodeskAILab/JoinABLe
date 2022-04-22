import torch
import matplotlib.pyplot as plt


def bounding_box_pointcloud(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


def bounding_box_uvsolid(inp):
    pts = inp[:, :, :, :3].reshape((-1, 3))
    mask = inp[:, :, :, 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)


def plot_uvsolid(uvsolid, ax, labels=None, points=True, normals=True, color=None):
    """Plot the UV solid features, where input is a of shape [#faces, #u, #v, 7]"""
    assert len(uvsolid.shape) == 4  # faces x #u x #v x 7
    bbox = bounding_box_uvsolid(uvsolid)
    label_color_map = {
        0: color,       # Non-joint
        1: "green",     # Joint
        2: "red",       # Ambiguous
        3: "blue",      # Joint Equivalent
        4: "orange",    # Ambiguous Equivalent
        5: "purple",    # Hole
        6: "pink"       # Hole Equivalent
    }
    if label_color_map[0] is None:
        label_color_map[0] = "gray"
    original_color = label_color_map[0]
    num_faces = uvsolid.size(0)
    for i in range(num_faces):
        pts = uvsolid[i, :, :, :3].cpu().detach().numpy().reshape((-1, 3))
        nor = uvsolid[i, :, :, 3:6].cpu().detach().numpy().reshape((-1, 3))
        mask = uvsolid[i, :, :, 6].cpu().detach().numpy().reshape(-1)
        point_indices_inside_faces = mask == 1
        pts = pts[point_indices_inside_faces, :]
        if labels is not None and i in labels:
            color = label_color_map[labels[i]]
        else:
            color = original_color
        if normals:
            nor = nor[point_indices_inside_faces, :]
            ax.quiver(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                nor[:, 0],
                nor[:, 1],
                nor[:, 2],
                length=0.15
            )
        if points:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color)
            ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])


def plot_point_cloud(pcd1, pcd2=None, normals1=None, normals2=None, title=None, ax=None, c1="green", c2="blue"):
    """Plot a point cloud or two

    :param pcd1: Point cloud with size (num_points, 3)
    :type pcd1: tensor
    :param pcd2: Point cloud with size (num_points, 3), defaults to None
    :type pcd2: tensor, optional
    :param normals1: Normal direction for each point with size (num_points, 3), defaults to None
    :type normals1: tensor
    :param normals2: Normal direction for each point with size (num_points, 3), defaults to None
    :type normals2: tensor
    :param title: Title for the plot, defaults to None
    :type title: string, optional
    :param ax: Matplotlib ax to draw into
    :type title: ax, optional
    """
    using_ax = ax is not None
    if not using_ax:
        fig = plt.figure(figsize=plt.figaspect(1.0))
        if title:
            fig.suptitle(title, fontsize=16)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(pcd1[:, 0], pcd1[:, 1], pcd1[:, 2], c=c1)
    if pcd2 is not None:
        ax.scatter(pcd2[:, 0], pcd2[:, 1], pcd2[:, 2], c=c2)
    if normals1 is not None:
        ax.quiver(
            pcd1[:, 0],
            pcd1[:, 1],
            pcd1[:, 2],
            normals1[:, 0],
            normals1[:, 1],
            normals1[:, 2],
            length=0.15
        )
    if normals2 is not None and pcd2 is not None:
        ax.quiver(
            pcd2[:, 0],
            pcd2[:, 1],
            pcd2[:, 2],
            normals2[:, 0],
            normals2[:, 1],
            normals2[:, 2],
            length=0.12
        )
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    if not using_ax:
        plt.show()