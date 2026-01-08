from script import local_density, find_nearest_corner
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import cv2


if __name__ == "__main__":
    # test the post processing function
    img = Image.open(r"C:\Users\Valentin\Documents\GIT_REPS\TUHH\ISM\Baselines\phase_2b\images\b00_i01_a01_20240813_152309_left_0003.jpg")
    box = [0.8412,  16.7431, 357.7286, 267.9004]    
    xmin, ymin, xmax, ymax = box
    
    xmin += 2; ymin += 2
    xmax -= 2; ymax -= 2

    # crop image
    cropped_img = img.crop((xmin, ymin, xmax, ymax))

    # pca on edges
    cropped_img = np.array(cropped_img)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 20, 200)
    ys, xs = np.where(edges > 0)

    edge_coords = np.column_stack([xs, ys])
    pca = PCA(n_components=1).fit(edge_coords)
    axis = pca.components_[0]

    # Project points onto axis
    edge_proj = edge_coords @ axis

    edge_order = np.argsort(edge_proj)
    edge_coords_sorted = edge_coords[edge_order]
    # compare corner brightness
    corner_size = 100
    end_1 = edge_coords_sorted[:corner_size]
    end_2 = edge_coords_sorted[-(corner_size+1):]
    mean_1 = np.mean([gray[int(p[1]), int(p[0])] for p in end_1])
    mean_2 = np.mean([gray[int(p[1]), int(p[0])] for p in end_2])

    tip_side = -1 if mean_1 < mean_2 else 0
    print("tip side:", tip_side)

    # find density transition
    r = int(0.2 * max(cropped_img.shape[:2]))  # 2% of size
    r = max(r, 5)

    densities = np.array([
        local_density(p, edges, r) for p in edge_coords_sorted
    ])

    i_transition = np.argmax(densities) # WIll this work?

    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else axis
    center = edge_coords_sorted.mean(axis=0)

    if tip_side == 0:
        tip_point = edge_coords_sorted[0]
        raw_transition_point = edge_coords_sorted[i_transition]
    else:
        tip_point = edge_coords_sorted[-1]
        raw_transition_point = edge_coords_sorted[-(i_transition+1)]

    # project the transition candidate onto the principal axis centered at 'center'
    vec = raw_transition_point - center
    proj_scalar = vec @ axis
    transition_point = center + proj_scalar * axis

    tip_corner = find_nearest_corner(tip_point, box)

    xmin = int(min(tip_corner[0], transition_point[0]))
    xmax = int(max(tip_corner[0], transition_point[0]))
    ymin = int(min(tip_corner[1], transition_point[1])) 
    ymax = int(max(tip_corner[1], transition_point[1]))

    tip_box = [xmin, ymin, xmax, ymax]
    
    # Visualize
    img_vis = cropped_img.copy()

    # AXIS___________________
    # ensure axis is a unit vector (axis is [ax,ay])
    axis = axis / np.linalg.norm(axis)

    # pick a center to position the axis: use the mean of edge points (x,y)
    center = edge_coords.mean(axis=0)   # (x, y) in crop coordinates

    # length to draw (pixels). use image diagonal to cover whole crop
    h, w = gray.shape[:2]
    length = int(1.5 * np.hypot(w, h))

    # endpoints in (x,y)
    p1 = (int(round(center[0] - axis[0] * length)), int(round(center[1] - axis[1] * length)))
    p2 = (int(round(center[0] + axis[0] * length)), int(round(center[1] + axis[1] * length)))

    # clamp to image bounds
    p1 = (max(0, min(w-1, p1[0])), max(0, min(h-1, p1[1])))
    p2 = (max(0, min(w-1, p2[0])), max(0, min(h-1, p2[1])))

    # draw
    cv2.line(img_vis, p1, p2, (0, 255, 255), 2)  # cyan axis
    # __________________

    print("xmin, ymin, xmax, ymax:", xmin, ymin, xmax, ymax)
    print("box", box)

    cv2.circle(img_vis, (int(tip_point[0]), int(tip_point[1])), 5, (0, 0, 255), -1)
    cv2.circle(img_vis, (int(tip_corner[0]), int(tip_corner[1])), 5, (255, 0, 255), -1)
    cv2.putText(img_vis, "tip corner", (int(tip_corner[0])+10, int(tip_corner[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.putText(img_vis, "tip point", (int(tip_point[0])+10, int(tip_point[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img_vis, "transition", (int(transition_point[0])+10, int(transition_point[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.circle(img_vis, (int(transition_point[0]), int(transition_point[1])), 5, (255, 0, 0), -1)
    cv2.rectangle(img_vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow("tip detection", img_vis)
    cv2.waitKey(0)