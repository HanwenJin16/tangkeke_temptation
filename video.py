import cv2 
import os
import numpy as np



# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture('input.mp4')

eye_cascade=cv2.CascadeClassifier("anime-eyes-cascade.xml")

def kanon_kuku(input,convert_param,sigma):
    input=cv2.cvtColor(input,cv2.COLOR_BGR2HSV)
    lower=np.array([104,75,62])
    upper=np.array([160,238,251])
    inputblurred=cv2.GaussianBlur(input,(11,11),sigmaX=sigma,sigmaY=sigma)
    mask=cv2.inRange(inputblurred,lower,upper).copy().astype("float")
    mask=mask/255
    notmask=1.0-mask
    input=input.copy().astype("float64")
    inputmask=np.zeros_like(input)
    inputmask[:,:,0]=input[:,:,0]*mask
    inputmask[:,:,1]=input[:,:,1]*mask 
    inputmask[:,:,2]=input[:,:,2]*mask
    inputnotmask=np.zeros_like(input)
    inputnotmask[:,:,0]=input[:,:,0]*notmask
    inputnotmask[:,:,1]=input[:,:,1]*notmask 
    inputnotmask[:,:,2]=input[:,:,2]*notmask
    inputmask[:,:,0]=((inputmask[:,:,0]-convert_param[0][0])*convert_param[1][0]+convert_param[2][0])*mask
    inputmask[:,:,1]=((inputmask[:,:,1]-convert_param[0][1])*convert_param[1][1]+convert_param[2][1])*mask
    inputmask[:,:,2]=((inputmask[:,:,2]-convert_param[0][2])*convert_param[1][2]+convert_param[2][2])*mask
    output=inputmask+inputnotmask
    return cv2.cvtColor(output.astype(np.uint8),cv2.COLOR_HSV2BGR)

convert_param=[[104,75,62],[1.31481481481,1.20552147239,0.687830687],[60,50,125]]

def process_boxes(boxes):
    if len(boxes)==0:
        return []
    else:
        new_box=[]
        for box in boxes:
            a=box[0]
            b=box[1]
            c=box[2]
            d=box[3]
            ai=int(a)
            bi=int(b)
            ci=int(c)
            di=int(d)
            new_box.append((ai,bi,ci,di))
        return new_box


if (vid_capture.isOpened() == False):
	print("Error opening the video file")
# Read fps and frame count
else:
	# Get frame rate information
	# You can replace 5 with CAP_PROP_FPS as well, they are enumerations
	fps = vid_capture.get(5)
	print('Frames per second : ', fps,'FPS')
	output_video= cv2.VideoWriter('output.mp4', -1, fps, (1920,1080))
	# Get frame count
	# You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
	frame_count = vid_capture.get(7)
	print('Frame count : ', frame_count)

while(vid_capture.isOpened()):
	# vid_capture.read() methods returns a tuple, first element is a bool 
	# and the second is frame
	ret, frame = vid_capture.read()
	if ret == True:
		eyes=eye_cascade.detectMultiScale(frame, scaleFactor = 1.2,minNeighbors = 4)
		eyes=process_boxes(eyes)

		for eye in eyes:
			x,y,w,h=eye
			eye_pixels=frame[y:y+h,x:x+w,:]
			eye_pixels=kanon_kuku(eye_pixels,convert_param,10)
			frame[y:y+h,x:x+w,:]=eye_pixels
		

		cv2.imshow('Frame',frame)
		output_video.write(frame)
		# 20 is in milliseconds, try to increase the value, say 50 and observe
		key = cv2.waitKey(1)
		
		if key == ord('q'):
			break
	else:
		break

# Release the video capture object
vid_capture.release()
output_video.release()
cv2.destroyAllWindows()
#cv2.VideoWriter('output.avi',fourcc, 20.0, (1920,1080))