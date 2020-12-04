Facial Expression recognition using CNN

=> Requirement.txt
tensorflow==2.2.0
opencv-python==4.2.0.34
pandas==1.1.1
numpy==1.19.1
scikit-learn==0.23.2

=>Workflow

1)Dataset and Transformation
total 2732 48 x 48 gray scale images of total 7 types of expression (Happy,Sad,Disgust,Angry,Neutral,Surprised,Fear) are transformed from 3 channeled.
JPG to single channel and stored in form of .XLSX
( Image dataset taken from kaggle. )

2)Training
Values of images are fatched from excel sheet and converted into original numpy of 48 x 48. CNN model is trained by the numpy array of images.

3)Detection
From video stream face is detected using pre-trained 'haarcascade_frontalface_default' and detected image is to trained CNN model and result is deplayed in the frame.  
 

