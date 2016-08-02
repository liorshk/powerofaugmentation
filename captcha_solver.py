import cv2
import numpy as np
from matplotlib import pyplot as plt

def predict_captcha(lettersModel,imgPath,displayLetters = True,numOfLetters = 6):
    """ Given an image it splits it into 'numOfLetters' and predicts the letter for each part """

    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    im = 255-img

    imgs = np.zeros((numOfLetters, 1, 32, 32), dtype=np.uint8)
    t = np.floor(im.shape[1] / float(numOfLetters))
    dd = 0
    bb = np.zeros((im.shape[0], dd), dtype=np.uint8) + 255

    im1 = im.transpose()[0:int(np.floor(t) + dd)].transpose()
    imgs[0, 0] = cv2.resize(np.concatenate((im1, bb), axis=1), (32, 32))
    for i in range(1,numOfLetters-1):
        from_pix  = int(np.floor(i*t) - dd)
        to_pix  = int(np.floor((i+1)*t) - dd)
        imi = im.transpose()[from_pix:to_pix].transpose()
        imgs[i, 0] = cv2.resize(imi, (32, 32))

    im_end = im.transpose()[int(np.floor((numOfLetters-1) * t) - dd):].transpose()
    imgs[numOfLetters-1, 0] = cv2.resize(np.concatenate((im_end, bb), axis=1), (32, 32))

    if displayLetters:
        fig, ax = plt.subplots(1,numOfLetters,figsize=(10,10))
        for i in range(0,numOfLetters):
            ax[i].imshow(imgs[i,0])
        plt.show()

    imgs = imgs.astype('float32') / 255.0
    classes = lettersModel.getModel().predict_classes(imgs, verbose=0)
    result = []
    for c in classes:
        result.append(lettersModel.getLetters()[c])

    prediction = ''.join(result).upper()

    return prediction

