import cv2
from ultralytics import YOLO, solutions

path_model = 'yolov8m.pt'
input_vid = 'enter your path'
output_vid = 'output.avi'
#init
model = YOLO(path_model)

#initalize Object Conter.

counter = solutions.ObjectCounter(
    view_img= True,
    reg_pts= [(350, 40), (1080, 40), (1080, 360), (350, 360)],
    classes_names=model.names,
    draw_tracks= True,
    line_thickness= 2
)

cap = cv2.VideoCapture(input_vid)
assert cap.isOpened() , "Error in input"

video_write = cv2.VideoWriter(output_vid, cv2.VideoWriter_fourcc(*"mp4v"),30,(1080,1920))

frame_count = 0  
while cap.isOpened():
    success , frame = cap.read()
    if not success:
        print("something Wrong")
        break


    tracks = model.track(frame , persist= True , iou = 0.2) # this make the object tracking happen botSort is for multi-trackng at the cost of computational power

    frame = counter.start_counting(frame , tracks)
    video_write.write(frame)
    frame_count += 1
     # Display the frame
    #cv2.imshow('Frame', frame)

    # Break the loop when '.q' key is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
video_write.release()
cv2.destroyAllWindows()
