import cv2
import time

# FOR CAMERA USAGES
#===================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# -- FOR FPS SETTINGS --
frame_count = 0
fps_display = 0.0
window_start = time.time()
window_sec = 0.1  # Update 10 times per second (Smaller value -> Faster FPS changes)

while True:
    ret, frame = cap.read()

    # CANNOT READ FRAME EXCEPTION
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
    # FPS
    font = cv2.FONT_HERSHEY_COMPLEX

    frame_count += 1
    now = time.time()
    elapsed = now - window_start

    if elapsed >= window_sec:
        fps_display = frame_count / elapsed
        frame_count = 0
        window_start = now

    fps = f"FPS: {fps_display:.1f}"

    cv2.putText(frame,       # Name of the frame
                fps,         # The FPS string
                (7,50),      # Location (x, y)
                1,           # Font scale
                font,        # Font style
                (100,250,0), # Text color (B,G,R) (0-255,0-255,0-255)
                3)           # Font Thickness


    # - SHOW FRAME
    cv2.imshow('frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# FOR VIDEOS
# -------------------------------------------------------
# cap = cv2.VideoCapture('vtest.avi')
 
# while cap.isOpened():
#     ret, frame = cap.read()
 
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) == ord('q'):
#         break
 
# cap.release()
# cv2.destroyAllWindows()