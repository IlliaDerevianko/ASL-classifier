import cv2
import os


def preprocess(img):
    roi = frame[height // 2 - 200:height // 2 + 200, width // 2 - 200:width // 2 + 200]
    roi = cv2.resize(roi, (100, 100))
    return roi

dataset_path = 'test_set'
capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
current_class = 'del'
if not os.path.isdir(os.path.join(dataset_path, current_class)):
    os.makedirs(os.path.join(dataset_path, current_class)) 
sub_dirs = [direc for direc in os.listdir(os.path.join(dataset_path, current_class)) if not direc.startswith('.')]
img_counter = len(sub_dirs)
while True:
    ret, frame = capture.read()
    if not ret:
        print("failed to grab frame")
        break
    prompt = f"Pictures of '{current_class}' taken: {img_counter}"
    cv2.putText(frame, prompt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.rectangle(frame, (width // 2 - 200, height // 2 - 200), (width // 2 + 200, height // 2 + 200), (0, 255, 0), 2)
    cv2.imshow("test", frame)
    frame = preprocess(frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = f"{current_class}_{img_counter}_test.png"
        path = os.path.join(dataset_path, current_class, img_name)
        print(path)
        cv2.imwrite(path, frame)
        print("{} written!".format(img_name))
        img_counter += 1

capture.release()

cv2.destroyAllWindows()