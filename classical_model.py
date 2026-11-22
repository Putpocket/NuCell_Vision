"""고전 기법 담당: Otsu + 거리 변환 + 마커 기반 Watershed.

입력은 팀 U-Net/Mask R-CNN과 같은 RGB uint8 [H,W,3] 배열입니다.
"""
import time
import cv2
import numpy as np

PARAMETER_COUNT = 0  # 학습하는 가중치가 없음. 알고리즘 설정값과 구분.

def resize_input(image, size):
    """Same resize/zero-padding convention as train_unet_improved.py."""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=method)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, (top, left, nh, nw, h, w)


def compact_labels(labels, min_area):
    """Remove small instances and number the remaining instances from one."""
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep != 0]
    lut = np.zeros(len(areas), np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[labels]


def segment_classical(image, polarity="auto", size=512, seed_ratio=0.7, min_area=16):
    """Return original-resolution labels and diagnostic stages; never read GT."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 image [H,W,3].")
    if min(image.shape[:2]) < 1 or size < 16 or not 0 < seed_ratio < 1 or min_area < 1:
        raise ValueError("Invalid image dimensions or classical parameters.")
    if polarity not in ("bright", "auto"):
        raise ValueError("polarity must be bright or auto.")
    canvas, geometry = resize_input(image, size)
    top, left, nh, nw, h, w = geometry
    valid = np.zeros((size, size), np.uint8)
    valid[top:top + nh, left:left + nw] = 255

    # Exclude padding from histogram and polarity estimation.
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    patch = gray[top:top + nh, left:left + nw]
    smooth = cv2.GaussianBlur(patch, (3, 3), 0)
    threshold, otsu_patch = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate((otsu_patch[0], otsu_patch[-1], otsu_patch[1:-1, 0], otsu_patch[1:-1, -1]))
    inverted = polarity == "auto" and float(np.mean(border > 0)) > 0.5
    if inverted:
        otsu_patch = cv2.bitwise_not(otsu_patch)
    otsu = np.zeros_like(gray)
    otsu[top:top + nh, left:left + nw] = otsu_patch

    # Morphological opening removes isolated specks before distance transform.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    clean = cv2.bitwise_and(clean, valid)
    _, components, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    kept = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_area)
    kept = kept[kept != 0]
    clean = np.where(np.isin(components, kept), 255, 0).astype(np.uint8)

    # Add an external background rim so edge-touching objects have finite distance.
    padded = cv2.copyMakeBorder(clean, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]

    # Threshold distance separately per connected component. This protects smaller
    # isolated nuclei when the image contains a few very large nuclei.
    _, components, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    seeds = np.zeros_like(clean)
    for component in range(1, len(stats)):
        x, y, cw, ch = stats[component, :4]
        region = components[y:y + ch, x:x + cw] == component
        local = distance[y:y + ch, x:x + cw]
        selected = region & (local > seed_ratio * local[region].max())
        seeds[y:y + ch, x:x + cw][selected] = 255

    seed_count, seed_labels = cv2.connectedComponents(seeds, connectivity=8)
    sure_region = cv2.dilate(clean, kernel, iterations=2)
    markers = seed_labels.astype(np.int32) + 1  # background=1, nucleus seeds>=2
    markers[(sure_region > 0) & (seeds == 0)] = 0  # unknown
    markers[valid == 0] = 1
    # OpenCV floods the image gradient from these distance-derived markers.
    if seed_count > 1:
        markers = cv2.watershed(canvas.copy(), markers)
        labels = np.where((markers > 1) & (clean > 0), markers - 1, 0).astype(np.int32)
        labels = compact_labels(labels, min_area)
    else:
        labels = np.zeros((size, size), np.int32)

    crop = labels[top:top + nh, left:left + nw]
    # NEAREST preserves instance IDs. A second compacting removes IDs lost when
    # reducing an upscaled small source image back to its original dimensions.
    restored = cv2.resize(crop.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST_EXACT).astype(np.int32)
    restored = compact_labels(restored, 1)
    stages = {"otsu": otsu, "distance": distance, "seeds": seeds,
              "labels_512": labels, "geometry": geometry,
              "threshold": float(threshold), "inverted": bool(inverted),
              "seed_count": int(seed_count - 1)}
    return restored, stages

def predict_classical(image, *, polarity="auto", size=512, seed_ratio=0.7, min_area=16):
    """평가 담당자용 인터페이스: binary(bool), predicted_count(int), time_ms(float).

    시간 범위: 메모리의 RGB 입력 -> 전처리 -> 분할 -> 원본 크기 복원 -> 개수 계산.
    파일 입출력과 정답/지표 계산은 포함하지 않습니다. 평가자는 사전 워밍업 후 호출합니다.
    """
    started = time.perf_counter()
    labels, _ = segment_classical(image, polarity=polarity, size=size,
                                 seed_ratio=seed_ratio, min_area=min_area)
    binary = labels > 0
    predicted_count = int(labels.max())
    elapsed_ms = (time.perf_counter() - started) * 1000
    return binary, predicted_count, elapsed_ms
