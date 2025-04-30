import os
import glob
import cv2

local_path = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats'
save_path = '/home/est_posgrado_manuel.suarez/homeworks/homework-ml2-extract-cats-dogs-faces'
files = glob.glob(os.path.join(local_path, 'train', '*.*.jpg'))
print(len(files))

c = 20
faceCascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
for file in files:
    img_bgr = cv2.imread(file, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detectedFaces = faceCascade.detectMultiScale(img_rgb, 1.1, 4)
    print(file, detectedFaces)
    # for (length,breadth,width,height) in detectedFaces:
    for (x, y, w, h) in detectedFaces:
        # cv2.rectangle(img_rgb, (x - c - 1, y - c - 1), (x + w + c + 1, y + h + c + 1), (255, 0, 0), 1)
        (path, ext) = os.path.splitext(file)
        name = path.split("/")[-1]
        # print(path, name, ext)
        # extract and save the ROI
        fname = os.path.join(save_path, 'files', name + '.face' + ext)
        print(fname)
        if x > c:
            x1 = x - c
        else:
            x1 = x
        if y > c:
            y1 = y - c
        else:
            y1 = y
        cv2.imwrite(fname, img_rgb[y1:y + h + c, x1:x + w + c])