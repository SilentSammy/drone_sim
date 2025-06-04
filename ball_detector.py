import cv2
import numpy as np

class BallDetector:
    def __init__(
        self,
        low_threshold=100,
        high_threshold=200,
        dilate_kernel_size=(5, 5),
        min_area=200
    ):
        """
        Initializes the BallDetector with optional custom parameters.

        Args:
            low_threshold (int): lower threshold for Canny edge detection.
            high_threshold (int): upper threshold for Canny edge detection.
            dilate_kernel_size (tuple): kernel size for dilating edges.
            min_area (int): minimum contour area to keep.
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.dilate_kernel_size = dilate_kernel_size
        self.min_area = min_area

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1: Convert to grayscale.
    # ──────────────────────────────────────────────────────────────────────────
    def to_gray(self, frame, drawing_frame=None):
        """
        Converts a BGR (or already grayscale) frame to single-channel grayscale.
        If drawing_frame is provided, visualizes the grayscale image (converted back to BGR).

        Returns:
            gray (ndarray): single-channel grayscale image.
        """
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return gray

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2: Run Canny on the grayscale image.
    # ──────────────────────────────────────────────────────────────────────────
    def get_edges(self, frame, drawing_frame=None):
        """
        Applies Canny edge detection to the grayscale version of frame.
        If drawing_frame is provided, visualizes the edges as a BGR image.

        Returns:
            edges (ndarray): binary edge map from Canny.
        """
        gray = self.to_gray(frame, drawing_frame=None)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)

        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return edges

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3: Dilate the Canny edges.
    # ──────────────────────────────────────────────────────────────────────────
    def dilate_edges(self, frame, drawing_frame=None):
        """
        Dilates the Canny edges using a rectangular kernel of size dilate_kernel_size.
        If drawing_frame is provided, visualizes the dilated edges as a BGR image.

        Returns:
            dilated (ndarray): binary image of dilated edges.
        """
        edges = self.get_edges(frame, drawing_frame=None)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.dilate_kernel_size)
        dilated = cv2.dilate(edges, kernel)

        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

        return dilated

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 4: Find black (non-edge) contours on the inverted dilated-edge mask.
    # ──────────────────────────────────────────────────────────────────────────
    def find_black_contours(self, frame, drawing_frame=None):
        """
        Finds contours corresponding to black regions (holes) in the inverted dilated-edge mask.
        Filters out contours with area < min_area.
        If drawing_frame is provided, draws filtered contours in blue on a black canvas.

        Returns:
            filtered (list of ndarray): list of contours (each a numpy array of points).
        """
        dilated = self.dilate_edges(frame, drawing_frame=None)
        inv = cv2.bitwise_not(dilated)
        _, binary = cv2.threshold(inv, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_area:
                filtered.append(cnt)

        if drawing_frame is not None:
            drawing_frame[:] = 0
            cv2.drawContours(drawing_frame, filtered, -1, (255, 0, 0), 2)

        return filtered

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 5: Fit ellipses to each filtered contour.
    # Returns a list of (contour, ellipse) pairs.
    # ──────────────────────────────────────────────────────────────────────────
    def fit_ellipses(self, frame, drawing_frame=None):
        """
        Fits an ellipse to each contour found in Stage 4 that has at least 5 points.
        If drawing_frame is provided, draws all fitted ellipses in green on a black canvas.

        Returns:
            contour_ellipse_pairs (list of tuples): each tuple is (contour, ellipse_params),
                where ellipse_params = ((cx, cy), (MA, ma), angle).
        """
        contours = self.find_black_contours(frame, drawing_frame=None)
        contour_ellipse_pairs = []

        for cnt in contours:
            if len(cnt) < 5:
                continue
            ellipse = cv2.fitEllipse(cnt)
            contour_ellipse_pairs.append((cnt, ellipse))

        if drawing_frame is not None:
            drawing_frame[:] = 0
            for _, ellipse in contour_ellipse_pairs:
                cv2.ellipse(drawing_frame, ellipse, (0, 255, 0), 2)

        return contour_ellipse_pairs

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 6: Choose the best ellipse based on circularity and area fit.
    # ──────────────────────────────────────────────────────────────────────────
    def find_best_ellipse(self, frame, drawing_frame=None):
        """
        Chooses the best ellipse among those found in Stage 5 using a combined score:
            score = normalized area error + normalized axis error.

        If drawing_frame is provided, overlays the chosen ellipse (in red) and center (in yellow)
        on top of the original frame (drawing_frame is assumed to be a copy of frame).

        Returns:
            best_ellipse (tuple): ellipse parameters ((cx, cy), (MA, ma), angle), or None if none found.
        """
        pairs = self.fit_ellipses(frame, drawing_frame=None)

        best_pair = None
        best_score = float("inf")

        for cnt, ellipse in pairs:
            (cx, cy), (MA, ma), angle = ellipse
            contour_area = cv2.contourArea(cnt)
            ellipse_area = np.pi * (MA / 2) * (ma / 2)

            # Area‐fit error (normalized)
            area_error = abs(ellipse_area - contour_area) / ellipse_area
            # Axis (circularity) error (normalized)
            axis_error = abs(MA - ma) / max(MA, ma)

            score = area_error + axis_error
            if score < best_score:
                best_score = score
                best_pair = (cnt, ellipse)

        if best_pair is None:
            return None

        cnt, best_ellipse = best_pair
        (cx, cy), (MA, ma), angle = best_ellipse

        if drawing_frame is not None:
            # Overlay on the original frame; do NOT clear drawing_frame
            cv2.ellipse(drawing_frame, best_ellipse, (0, 0, 255), 2)
            cv2.circle(drawing_frame, (int(cx), int(cy)), 2, (0, 255, 255), -1)

        return best_ellipse
