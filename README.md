This program detects faces in images and determines whether each face is wearing a mask or not.

To classify faces in your photos, follow these simple steps:
- open a terminal in the main directory, called 'mask_detection'
- to download all the module dependencies, run: pip install -r requirements.txt
- place the pictures you wish to have classified, in the 'masks' folder 
- to run the program, run the following command: python predicting.py
- now, you can retrieve the classified faces, from the 'detected' folder

Note :  Faces "With" and "Without" a mask, will be flagged in their file name at the beginning, so you can search the
        flag in your directory to copy all files with a certain flag

Note :  This mask detection model has been trained on this dataset  --> https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
        It's a CNN with 90% accuracy       