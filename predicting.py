import tensorflow as tf
import cv2 as cv
import numpy as np
import os
import mediapipe as mp

IMG_WIDTH = 30
IMG_HEIGHT = 30

FACE_NUM = 15

classes = ['Without', 'With']

#a list of the face mesh points, making up the face outline polygon 
outer_points = [10, 338, 297, 332, 284, 447, 288, 365, 378, 400, 152, 148, 176, 149, 150, 136, 172, 215, 177, 137,
                162, 21, 54, 103, 67, 109]

def get_faces(img):
    '''
    Takes in an ndarray image 'img', detects all the faces, returns a list of rectangle ndarray face crops of 'img'
    '''
    rgb = cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB)

    #initializing face mesh detection objects
    mp_mesh = mp.solutions.face_mesh
    mesh = mp_mesh.FaceMesh(max_num_faces=FACE_NUM, min_detection_confidence=0.25)

    #finding all face meshes in the RGB image
    results = mesh.process(rgb)

    h, w, c = img.shape

    #the list to contain the lists of faces, containing landmark x-y coordinates in them
    faces = []
    #if faces exist, iterate over them
    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            face = []
            #iterating over landmarks in each individual face
            face_lms_ls = face_lms.landmark
            for i in outer_points:
                #appending the de-normalized landmark x-y coordinates list to the face list
                face.append([int(w*face_lms_ls[i].x), int(h*face_lms_ls[i].y)])
            #adding the face list to the list of faces
            faces.append(face)


    #creating the mask, then drawing the face outline polygons, face by face
    images = []      
    mask = np.zeros_like(img)
    for face in faces:
        arr = np.int32([face])
        cv.polylines(mask, pts=arr, isClosed=True, color=(255, 255, 255), thickness=1)

    #convert the mask to one grayscale color channel and finding its contour coordinates
    src_gray = cv.cvtColor(mask.copy(), cv.COLOR_BGR2GRAY)
    
    contours, _ = cv.findContours(src_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    for i in range(len(contours)):
        buff = img[int(boundRect[i][1]):int(boundRect[i][1]+boundRect[i][3]), int(boundRect[i][0]):int(boundRect[i][0]+boundRect[i][2])]
        images.append(buff)

    return images


#loading the 'mask_model' as 'model'
model = tf.keras.models.load_model('mask_model')

#loading images
images_str = os.listdir('masks')
for image_str in images_str:
    #formatting the image for use in 'model', then doing the prediction
    
    im = cv.imread(os.path.join('masks', image_str))

    face_crops = get_faces(im)

    for i, face_crop in enumerate(face_crops):
        gray = cv.cvtColor(face_crop.copy(), code=cv.COLOR_BGR2GRAY)
        formatted = cv.resize(gray, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_CUBIC)/255
        predictions = model.predict(np.array([formatted]))
        result_text = classes[np.argmax(predictions[0])]
        
        cv.imwrite(os.path.join('detected', result_text + f'_{i}_' + image_str), face_crop) 

    

    
    

