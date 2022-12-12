import cv2
import os

video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

count=0

nameID=str(input("Enter Your Name and ID: ")).lower()

path='images/'+nameID

isExist = os.path.exists(path)

if isExist:
	print("Name Already Taken")
	nameID=str(input("Enter Your Name Again: "))
else:
	os.makedirs(path)

while True:
	ret,frame=video.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces=facedetect.detectMultiScale(gray,1.3, 5)
	for x,y,w,h in faces:
		count=count+1
		name='./Images/'+nameID+'/'+ str(count) + '.jpg'
		print("Creating Images........." +name)
		cv2.imwrite(name, gray [y:y+h,x:x+w]) #grayscale
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	cv2.imshow("Collecting Images", frame)
	cv2.waitKey(1)
	if count>30:
		break
video.release()
cv2.destroyAllWindows()
