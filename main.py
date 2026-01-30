import cv2

# FOR CAMERA
#===================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    cv2.rectangle(frame, (5, 5), (220, 220), (255, 0, 0), 2)
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