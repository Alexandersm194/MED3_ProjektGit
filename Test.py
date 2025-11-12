import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def rotate_image(img, angle_deg, borderMode=cv2.BORDER_CONSTANT):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=0)

def circular_mean_deg(angles_deg):
    """Wrap-safe mean of angles (deg). Returns (-180,180]."""
    if len(angles_deg) == 0:
        return None
    r = np.deg2rad(angles_deg)
    s = np.sin(r).mean()
    c = np.cos(r).mean()
    mean_rad = np.arctan2(s, c)
    mean_deg = np.rad2deg(mean_rad)
    # normalize to (-180,180]
    if mean_deg <= -180:
        mean_deg += 360
    if mean_deg > 180:
        mean_deg -= 360
    return mean_deg

# ---------- SVD (PCA) based on silhouette ----------
def SVD_fix(mask_image, visualize=True):
    """
    mask_image : binary (0/255) or grayscale mask where foreground pixels are nonzero
    Rotates so the major axis (PCA first component) becomes vertical.
    Returns (upright_mask, dominant_angle_deg, correction_deg)
    """
    # ensure binary / single channel
    if len(mask_image.shape) == 3:
        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_image.copy()

    ys, xs = np.nonzero(gray)
    if len(xs) < 10:
        print("Not enough mask pixels for SVD.")
        return mask_image, None, None

    pts = np.column_stack((xs, ys)).astype(np.float64)
    pts_centered = pts - pts.mean(axis=0)

    # small numerical stable covariance 2x2
    cov = np.cov(pts_centered, rowvar=False)
    _, s, Vt = np.linalg.svd(cov)  # Vt[0] is principal direction
    principal = Vt[0]  # unit vector (vx, vy)

    # Because OpenCV y axis points down, flip y for angle computation so result is in standard math coords
    dominant_angle = np.degrees(np.arctan2(-principal[1], principal[0]))  # signed in (-180,180]
    dominant_angle = (dominant_angle + 180) % 360 - 180  # normalize to (-180,180]

    # We want major axis vertical -> target angle = +/-90. Choose correction to nearest vertical orientation.
    # compute correction so that principal vector points upwards (i.e., rotate so angle becomes -90 or +90)
    # simpler: rotate so principal becomes +90 (vertical pointing right->up)
    # compute two candidate corrections: rotate by (90 - dominant_angle) or by (-90 - dominant_angle) and pick smaller absolute rotation
    cand1 = 90 - dominant_angle
    cand2 = -90 - dominant_angle
    correction = cand1 if abs(cand1) <= abs(cand2) else cand2

    upright = rotate_image(gray, correction, borderMode=cv2.BORDER_CONSTANT)

    if visualize:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.imshow(gray, cmap='gray'); plt.title(f"Before (angle={dominant_angle:.1f}°)"); plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(upright, cmap='gray'); plt.title(f"After correction ({correction:.1f}°)"); plt.axis('off')
        plt.show()
    return upright, dominant_angle, correction

# ---------- Hough-lines based (robust) ----------
def dominant_angle_from_lines_fix(bgr_or_gray_image,
                                  canny1=50, canny2=150,
                                  hough_thresh=60, minLineLength=40, maxLineGap=20,
                                  min_length_keep=20, cluster_width_deg=10, visualize=True):
    """
    Detect lines, compute signed-dominant angle (no mod 180), rotate back by -angle (or to nearest vertical).
    Returns (upright_image, chosen_angle_deg, debug_vis)
    """
    # grayscale
    if bgr_or_gray_image is None:
        raise ValueError("Empty image")
    if len(bgr_or_gray_image.shape) == 3:
        gray = cv2.cvtColor(bgr_or_gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr_or_gray_image.copy()

    gray_blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray_blur, canny1, canny2)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=hough_thresh,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is None or len(lines) == 0:
        print("No Hough lines found. Visualize edges to tune thresholds.")
        if visualize:
            plt.figure(figsize=(6,6)); plt.imshow(edges, cmap='gray'); plt.title("Edges"); plt.axis('off'); plt.show()
        return bgr_or_gray_image, None, edges

    # collect signed angles for sufficiently long segments
    angles = []
    filtered = []
    for ln in lines:
        x1,y1,x2,y2 = ln[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        if length < min_length_keep:
            continue
        angle = np.degrees(np.arctan2(dy, dx))  # signed angle in (-180,180]
        angles.append(angle)
        filtered.append((x1,y1,x2,y2,angle,length))

    if len(angles) == 0:
        print("No lines kept after length filter.")
        return bgr_or_gray_image, None, edges

    # circular mean baseline
    mean_ang = circular_mean_deg(angles)

    # histogram peak clustering to avoid averaging orthogonal structures
    angs = np.array(angles)
    # wrap to (-180,180]
    angs_wrapped = (angs + 180) % 360 - 180
    hist_bins = 360
    hist, bin_edges = np.histogram(angs_wrapped, bins=hist_bins, range=(-180,180))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_idx = np.argmax(hist)
    peak_center = bin_centers[peak_idx]

    # collect angles within cluster_width_deg of peak (wrap safe)
    def angle_diff(a,b):
        d = a - b
        d = (d + 180) % 360 - 180
        return d
    cluster = [a for a in angs_wrapped if abs(angle_diff(a, peak_center)) <= cluster_width_deg]

    chosen_angle = None
    if len(cluster) >= 3:
        chosen_angle = circular_mean_deg(cluster)
        method = f"cluster(size={len(cluster)})"
    else:
        chosen_angle = mean_ang
        method = "circular_mean"

    # Normalize chosen angle to (-180,180]
    if chosen_angle is None:
        return bgr_or_gray_image, None, edges
    chosen_angle = (chosen_angle + 180) % 360 - 180

    # We want to rotate so the dominant direction becomes vertical.
    # Option A: simply undo rotation so edges align as originally (rotate by -chosen_angle)
    correction = -chosen_angle
    # Option B: rotate to nearest vertical (uncomment if desired)
    # cand1 = 90 - chosen_angle
    # cand2 = -90 - chosen_angle
    # correction = cand1 if abs(cand1) <= abs(cand2) else cand2

    upright = rotate_image(bgr_or_gray_image, correction)

    if visualize:
        # draw filtered lines and chosen vector
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x1,y1,x2,y2,a,l) in filtered:
            cv2.line(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        # centroid
        mids_x = [(x1+x2)/2 for (x1,y1,x2,y2,_,_) in filtered]
        mids_y = [(y1+y2)/2 for (x1,y1,y2,y2,_,_) in filtered] if False else [(y1+y2)/2 for (x1,y1,x2,y2,_,_) in filtered]  # safe compute
        cx = int(np.mean(mids_x)); cy = int(np.mean(mids_y))
        rad = np.deg2rad(chosen_angle)
        length = max(vis.shape[:2]) * 0.25
        vx = int(np.cos(rad) * length); vy = int(np.sin(rad) * length)
        cv2.arrowedLine(vis, (cx,cy), (cx+vx, cy+vy), (0,0,255), 3, tipLength=0.2)

        plt.figure(figsize=(14,5))
        plt.subplot(1,3,1); plt.title("Edges"); plt.imshow(edges, cmap='gray'); plt.axis('off')
        plt.subplot(1,3,2); plt.title(f"Lines & chosen angle {chosen_angle:.1f}° ({method})"); plt.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)); plt.axis('off')
        plt.subplot(1,3,3);
        if len(upright.shape)==3:
            plt.imshow(cv2.cvtColor(upright, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(upright, cmap='gray')
        plt.title(f"After rotation (corr={correction:.1f}°)"); plt.axis('off')
        plt.show()

    print(f"Chosen angle: {chosen_angle:.2f}° (method={method}), correction applied: {correction:.2f}°")
    return upright, chosen_angle, filtered

