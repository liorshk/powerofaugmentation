from LettersModel import LettersModel
from captcha_solver import predict_captcha

from os import listdir

letters = list('abcdefghijklmnopqrstuvwxyz')
lettersModel = LettersModel(letters,weights_path="letters.h5")
lettersModel.generate_dataset(generateNoise=True)
lettersModel.load_model()

# First training on "easy" augmentations
lettersModel.train(save_model_to_file=True,rotation_range = 10,width_shift_range=0.2,height_shift_range=0.2)

# Training again on harder examples
lettersModel.train(save_model_to_file=True,rotation_range = 20,width_shift_range=0.5,height_shift_range=0.7)


# Predicting the captchas in the captchas directory

img_dir = 'capcha_labeled/'

names = listdir(img_dir)
for name in sorted(names):
    # Predict the captcha (it contains 6 letters)
    prediction = predict_captcha(lettersModel,img_dir+name,numOfLetters = 6)
    actual = name.split('_')[1][:-4]

    if(actual == prediction):
        print("Great!")
    else:
        print(prediction)
