"""
main.py  (improved hand_boundary_poc)
- Real-time hand approach detection using classical CV only (no MediaPipe/OpenPose).
- Improvements:
  * fingertip detection with confidence (convexity defects + fallback farthest point)
  * centroid + fingertip fusion (use fingertip only when confident)
  * bounding-box overlap fail-safe (any hand overlap -> DANGER)
  * history-based smoothing (longer deque to avoid flicker)
  * danger padding & warning visualization
  * flashing DANGER overlay for visibility
  * optional on-screen HSV calibration: press 'c' while your hand is inside the small calibration box
Usage:
    python main.py
Keys:
    ESC - quit
    c   - capture HSV calibration from small green box in top-left
    r   - reset HSV calibration to defaults
"""

import cv2
import numpy as np
import time
from collections import deque

# ----------------- PARAMETERS (tune these) -----------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BLUR_K = 7
MIN_CONTOUR_AREA = 1500

# Default (broad) skin HSV range (you can calibrate at runtime with 'c')
SKIN_HSV_LOWER = np.array([0, 30, 60])
SKIN_HSV_UPPER = np.array([25, 200, 255])

BOX_W, BOX_H = 200, 140

# Thresholds are in pixels (distance from chosen hand point to nearest box edge)
SAFE_DIST = 140
WARNING_DIST = 70
DANGER_DIST = 18

# Pixel margins / tolerances
TOLERANCE = 14           # general tolerance around box used earlier
DANGER_PADDING = 8       # extra slack for danger detection
BBOX_PAD = 10            # bounding box pad for overlap check

# Smoothing / hysteresis
STATE_HISTORY_LEN = 9    # larger = less flicker
# Fingertip must be separated from centroid by at least this many pixels to be considered
FINGERTIP_MIN_SEPARATION = 8

# Fingertip defect confidence threshold (0..1)
FINGERTIP_CONF_THRESH = 0.28

# ----------------------------------------------------------

def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None
    return largest

def find_fingertip_with_confidence(contour):
    """
    Try convexity-defect based fingertip detection and return (x,y,confidence).
    If defects missing, fall back to farthest contour point from centroid with low confidence.
    Confidence ~ normalized defect depth (0..1).
    """
    if contour is None or len(contour) < 5:
        return None, 0.0

    # compute centroid
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = np.mean(contour.reshape(-1,2), axis=0).astype(int)

    # use convex hull indices for defects
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        # fallback: farthest point from centroid
        pts = contour.reshape(-1,2)
        dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        far_idx = np.argmax(dists)
        fingertip = tuple(pts[far_idx])
        return fingertip, 0.15

    defects = cv2.convexityDefects(contour, hull_idx)
    if defects is None:
        pts = contour.reshape(-1,2)
        dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        far_idx = np.argmax(dists)
        fingertip = tuple(pts[far_idx])
        return fingertip, 0.12

    # find the defect with the largest depth (likely between fingers); fingertip candidates are start/end points
    best_depth = 0
    candidate_points = []
    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        # depth is multiplied by 256 in OpenCV
        if depth > best_depth:
            best_depth = depth
        candidate_points.append((s, e, f, depth))

    # If best_depth is tiny, fallback
    if best_depth < 300:  # empirical threshold
        pts = contour.reshape(-1,2)
        dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        far_idx = np.argmax(dists)
        fingertip = tuple(pts[far_idx])
        return fingertip, min(0.2, best_depth/3000.0)

    # Build fingertip candidates from defect endpoints (s and e) and choose the one farthest from centroid
    pts = contour.reshape(-1,2)
    candidates = []
    for s, e, f, depth in candidate_points:
        start_pt = tuple(contour[s][0])
        end_pt = tuple(contour[e][0])
        candidates.append(start_pt)
        candidates.append(end_pt)
    # remove duplicates
    candidates = list({c: None for c in candidates}.keys())

    if not candidates:
        pts = contour.reshape(-1,2)
        dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        far_idx = np.argmax(dists)
        fingertip = tuple(pts[far_idx])
        return fingertip, min(0.4, best_depth/4000.0)

    # pick candidate farthest from centroid
    cand_arr = np.array(candidates)
    dists = np.linalg.norm(cand_arr - np.array([cx, cy]), axis=1)
    best_idx = np.argmax(dists)
    fingertip = tuple(candidates[best_idx])

    # normalize confidence between 0 and 1 using best_depth
    conf = float(min(1.0, best_depth / 3000.0))
    return fingertip, conf

def distance_point_to_rect(px, py, rx1, ry1, rx2, ry2):
    dx = max(rx1 - px, 0, px - rx2)
    dy = max(ry1 - py, 0, py - ry2)
    if dx == 0 and dy == 0:
        return 0.0
    return np.hypot(dx, dy)

def state_from_dist(d):
    if d <= (DANGER_DIST + DANGER_PADDING):
        return "DANGER"
    if d <= WARNING_DIST:
        return "WARNING"
    if d <= SAFE_DIST:
        return "WARNING"
    return "SAFE"

def draw_dashed_rect(img, p1, p2, color, thickness=1, dash_len=8):
    x1,y1 = p1; x2,y2 = p2
    # top
    for x in range(x1, x2, dash_len*2):
        cv2.line(img, (x, y1), (min(x+dash_len, x2), y1), color, thickness)
    # bottom
    for x in range(x1, x2, dash_len*2):
        cv2.line(img, (x, y2), (min(x+dash_len, x2), y2), color, thickness)
    # left
    for y in range(y1, y2, dash_len*2):
        cv2.line(img, (x1, y), (x1, min(y+dash_len, y2)), color, thickness)
    # right
    for y in range(y1, y2, dash_len*2):
        cv2.line(img, (x2, y), (x2, min(y+dash_len, y2)), color, thickness)

def calibrate_hsv_from_box(frame):
    """
    Read small top-left box and compute HSV mean and set skin bounds around it.
    Press 'c' while your hand is in the calibration box.
    """
    small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    # calibration box in top-left: 40x40
    bx, by, bw, bh = 8, 8, 40, 40
    roi = small[by:by+bh, bx:bx+bw]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = int(np.mean(hsv_roi[:,:,0]))
    s_mean = int(np.mean(hsv_roi[:,:,1]))
    v_mean = int(np.mean(hsv_roi[:,:,2]))
    # build bounds around mean with fixed deltas
    h_delta = 12
    s_delta = 40
    v_delta = 60
    lower = np.array([max(0, h_mean - h_delta), max(20, s_mean - s_delta), max(10, v_mean - v_delta)])
    upper = np.array([min(179, h_mean + h_delta), min(255, s_mean + s_delta), min(255, v_mean + v_delta)])
    return lower, upper

def main():
    global SKIN_HSV_LOWER, SKIN_HSV_UPPER
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # virtual rect center
    vcx, vcy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    rx1 = int(vcx - BOX_W // 2)
    ry1 = int(vcy - BOX_H // 2)
    rx2 = int(vcx + BOX_W // 2)
    ry2 = int(vcy + BOX_H // 2)

    state_history = deque(maxlen=STATE_HISTORY_LEN)
    fps_smooth = None
    prev_time = time.time()

    calibrated = False

    print("Running. Press 'c' to calibrate HSV using top-left box. Press 'r' to reset calibration. ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        blurred = cv2.GaussianBlur(small, (BLUR_K, BLUR_K), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # skin mask
        mask = cv2.inRange(hsv, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # find hand contour
        contour = get_largest_contour(mask)
        fingertip = None
        fingertip_conf = 0.0
        hand_point = None  # chosen point used for distance calculation
        overlap = False     # bounding box overlap fail-safe

        if contour is not None:
            # fingertip detection with confidence
            fingertip, fingertip_conf = find_fingertip_with_confidence(contour)
            if fingertip is not None:
                cv2.circle(small, fingertip, 7, (0,0,200), -1)  # red (fingertip display)

            # draw contour + hull
            cv2.drawContours(small, [contour], -1, (0,255,0), 2)
            hull = cv2.convexHull(contour)
            cv2.drawContours(small, [hull], -1, (0,180,0), 2)

            # centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx_cont = int(M['m10']/M['m00'])
                cy_cont = int(M['m01']/M['m00'])
            else:
                cx_cont, cy_cont = np.mean(contour.reshape(-1,2), axis=0).astype(int)
            cv2.circle(small, (cx_cont, cy_cont), 6, (255,0,0), -1)  # blue centroid

            # choose hand_point: prefer fingertip if confident & separated sufficiently
            if fingertip is not None and fingertip_conf >= FINGERTIP_CONF_THRESH:
                sep = np.linalg.norm(np.array(fingertip) - np.array([cx_cont, cy_cont]))
                if sep >= FINGERTIP_MIN_SEPARATION:
                    hand_point = (int(fingertip[0]), int(fingertip[1]))
                else:
                    # if fingertip too close to centroid, use centroid (better stability)
                    hand_point = (cx_cont, cy_cont)
            else:
                hand_point = (cx_cont, cy_cont)

            # bounding-box overlap fail-safe
            x, y, w, h = cv2.boundingRect(contour)
            hx1, hy1 = x - BBOX_PAD, y - BBOX_PAD
            hx2, hy2 = x + w + BBOX_PAD, y + h + BBOX_PAD
            # check overlap between hand bbox and virtual box (use non-tolerance rx1/ry1)
            overlap = not (hx2 < rx1 or hx1 > rx2 or hy2 < ry1 or hy1 > ry2)

            # for debug draw hand bbox
            cv2.rectangle(small, (hx1, hy1), (hx2, hy2), (120,255,120), 1)

        else:
            # no contour found
            hand_point = None

        # compute distance to padded rectangle (apply tolerance)
        rx1_t = rx1 - TOLERANCE
        ry1_t = ry1 - TOLERANCE
        rx2_t = rx2 + TOLERANCE
        ry2_t = ry2 + TOLERANCE

        if hand_point is not None:
            d = distance_point_to_rect(hand_point[0], hand_point[1], rx1_t, ry1_t, rx2_t, ry2_t)
            cv2.circle(small, hand_point, 5, (0,255,255), -1)  # yellow = chosen point
        else:
            d = 9999

        # raw state and smoothing via history voting
        raw_state = state_from_dist(d)
        state_history.append(raw_state)
        # majority vote (if enough history frames exist)
        if len(state_history) == STATE_HISTORY_LEN:
            voted = max(set(state_history), key=state_history.count)
            current_state = voted
        else:
            current_state = raw_state

        # override: if hand bbox overlaps the virtual box => DANGER
        if overlap:
            current_state = "DANGER"

        # Draw virtual box and an outer dashed warning rectangle for WARNING region
        color_box = (0,255,0) if current_state == "SAFE" else (0,165,255) if current_state == "WARNING" else (0,0,255)
        cv2.rectangle(small, (rx1, ry1), (rx2, ry2), color_box, 2)

        # draw dashed warning region (visualize warning band)
        rx1_warn = rx1 - WARNING_DIST
        ry1_warn = ry1 - WARNING_DIST
        rx2_warn = rx2 + WARNING_DIST
        ry2_warn = ry2 + WARNING_DIST
        draw_dashed_rect(small, (rx1_warn, ry1_warn), (rx2_warn, ry2_warn), (0,165,255), 1)

        # Overlay state text
        cv2.putText(small, f"State: {current_state}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_box, 2)

        # Flashing DANGER overlay (toggle every 0.5s)
        if current_state == "DANGER":
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(small, "DANGER DANGER", (50, FRAME_HEIGHT//2), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0,0,255), 6)
        elif current_state == "WARNING":
            cv2.putText(small, "WARNING", (50, FRAME_HEIGHT//2), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,165,255), 4)

        # show distance and fingertip confidence
        if d < 9998:
            cv2.putText(small, f"Dist: {int(d)} px", (10, FRAME_HEIGHT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(small, f"FPS: {0.0:.1f}", (FRAME_WIDTH-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)  # placeholder for update

        # show fingertip confidence (debug)
        if fingertip is not None:
            cv2.putText(small, f"FTConf:{fingertip_conf:.2f}", (FRAME_WIDTH-190, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,50), 2)

        # compute and show FPS properly
        t1 = time.time()
        fps = 1.0 / (t1 - prev_time) if (t1 - prev_time) > 0 else 0.0
        prev_time = t1
        if fps_smooth is None:
            fps_smooth = fps
        else:
            fps_smooth = fps_smooth * 0.85 + fps * 0.15
        cv2.putText(small, f"FPS: {fps_smooth:.1f}", (FRAME_WIDTH-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # show mask preview bottom-right
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_bgr, (160,120))
        small[FRAME_HEIGHT-120:FRAME_HEIGHT, FRAME_WIDTH-160:FRAME_WIDTH] = mask_small

        # draw small calibration box top-left
        cbx, cby, cbw, cbh = 8, 8, 40, 40
        cv2.rectangle(small, (cbx, cby), (cbx+cbw, cby+cbh), (0,255,0), 1)
        cv2.putText(small, "c:cal", (8, 8+cbh+14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        cv2.imshow("Hand Boundary POC", small)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('c'):
            # calibrate HSV using small top-left box
            lower, upper = calibrate_hsv_from_box(frame)
            SKIN_HSV_LOWER = lower
            SKIN_HSV_UPPER = upper
            calibrated = True
            print("Calibrated HSV:", SKIN_HSV_LOWER, SKIN_HSV_UPPER)
        elif key == ord('r'):
            # reset to defaults
            SKIN_HSV_LOWER = np.array([0, 30, 60])
            SKIN_HSV_UPPER = np.array([25, 200, 255])
            calibrated = False
            print("HSV reset to defaults")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
