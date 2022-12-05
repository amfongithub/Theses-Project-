import cv2
import sys

cascPath = sys.argv[0]
faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.flip(frame,0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        cropped_color = frame[y:y+h, x:x+w]
        cropped_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y+h, x:x+w]

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    #cv2.imshow('Grayscale', gray)
    cv2.imshow('Cropped', cropped_color)
    cv2.imshow('Cropped_Img', cropped_gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
