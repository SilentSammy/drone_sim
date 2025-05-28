import cv2
import os
import time

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
