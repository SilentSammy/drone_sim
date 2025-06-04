import cv2
import os
import time

class VideoPlayer:
    def __init__(self, frame_source):
        self.frame_source = frame_source
        self.frame_count = 0
        self._frame_idx = 0.0
        self.fps = 30  # Default FPS
        self._get_frame = None
        self.last_time = None
        self.dt = 0.0
        self.setup_video_source()
        self.first_time = True

    def show_frame(self, img, name, scale=1):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        if self.first_time:
            self.first_time = False
            cv2.resizeWindow(name, int(img.shape[1]*scale), int(img.shape[0]*scale))
            self.first_time = False
        cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    def get_frame(self, idx=None):
        if idx is None:
            idx = self.frame_idx
        return self._get_frame(idx)

    def step(self, step_size=1):
        self._frame_idx += step_size
        self._frame_idx = self._frame_idx % self.frame_count
    
    def time_step(self):
        self.dt = time.time() - self.last_time if self.last_time is not None else 0.0
        self.last_time = time.time()
        return self.dt

    def move(self, speed=1):
        self._frame_idx += speed * self.dt * self.fps
        self._frame_idx = self._frame_idx % self.frame_count

    @property
    def frame_idx(self):
        return int(self._frame_idx)

    def setup_video_source(self):
        # If frame_source is a cv2.VideoCapture object, use it directly
        if isinstance(self.frame_source, cv2.VideoCapture):
            cap = self.frame_source
            if not cap.isOpened():
                print("Error opening video file")
                exit(1)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total frames:", self.frame_count)
            
            def get_frame(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    print("Failed to get frame", idx)
                    return None
                return frame
            
            self._get_frame = get_frame
        # If frame_source is a folder, load images
        elif os.path.isdir(self.frame_source):
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_files = sorted([
                os.path.join(self.frame_source, f) 
                for f in os.listdir(self.frame_source) 
                if f.lower().endswith(image_extensions)
            ])
            self.frame_count = len(image_files)
            print("Total frames (images):", self.frame_count)
            
            def get_frame(idx):
                idx = int(idx)
                if idx < 0 or idx >= len(image_files):
                    print("Index out of bounds:", idx)
                    return None
                frame = cv2.imread(image_files[idx])
                if frame is None:
                    print("Failed to load image", image_files[idx])
                return frame
            
            self._get_frame = get_frame
        else:
            # Assume frame_source is a video file.
            cap = cv2.VideoCapture(self.frame_source)
            if not cap.isOpened():
                print("Error opening video file:", self.frame_source)
                exit(1)
            
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total frames:", self.frame_count)
            
            def get_frame(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    print("Failed to get frame", idx)
                    return None
                return frame
            
            self._get_frame = get_frame

# DISPLAY
def show_frame(img, name, scale=1):
    show_frame.first_time = show_frame.first_time if hasattr(show_frame, 'first_time') else True
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    if show_frame.first_time:
        cv2.resizeWindow(name, int(img.shape[1]*scale), int(img.shape[0]*scale))
        show_frame.first_time = False
    cv2.imshow(name, img)
    if cv2.waitKey(1) & 0xFF == 27:
        raise KeyboardInterrupt

# RECORDING
def screenshot(frame):
    import os

    # Static variables
    screenshot.last_time = screenshot.last_time if hasattr(screenshot, 'last_time') else None
    screenshot.dir_path = screenshot.dir_path if hasattr(screenshot, 'dir_path') else "./screenshots/"+time.strftime("%Y-%m-%d_%H-%M-%S")
    screenshot.count = screenshot.count if hasattr(screenshot, 'count') else 0

    # If less than n seconds have passed since the last screenshot, return
    if screenshot.last_time is not None and time.time() - screenshot.last_time < 1:
        return
    screenshot.last_time = time.time()

    # Make the directory if it doesn't exist
    os.makedirs(screenshot.dir_path, exist_ok=True)

    # Save the image
    filename = os.path.join(screenshot.dir_path, f"screenshot_{screenshot.count}.png")
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved: {filename}")

    # Increment the count
    screenshot.count += 1

def record(frame):
    """
    If frame is a valid image (numpy array), write it to the video file.
    If frame is None, release the VideoWriter if it exists.
    """
    # Close the video if frame is None
    if frame is None:
        if hasattr(record, "vw"):
            record.vw.release()
            print("Video recording closed.")
            del record.vw
        return

    # If VideoWriter hasn't been created yet, initialize it now
    if not hasattr(record, "vw"):
        fps = 30  # desired frame rate
        height, width = frame.shape[:2]
        
        # Create a directory for video output if needed
        record.dir_path = record.dir_path if hasattr(record, "dir_path") else "./videos"
        os.makedirs(record.dir_path, exist_ok=True)
        
        # Create a timestamped file name 
        filename = os.path.join(record.dir_path, time.strftime("output_%Y-%m-%d_%H-%M-%S.mp4"))
        
        # Choose a codec that works well (e.g., 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        record.vw = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        print(f"Video recording started: {filename}")
    
    # Append the frame to the video file
    record.vw.write(frame)
