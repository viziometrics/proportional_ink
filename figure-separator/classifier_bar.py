import keras
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import glob
import numpy as np 
import shutil

model = load_model('models/classifybars.h5')
test_path = glob.glob(r'./extracts/*')
#print(test_path)
test_files = [f for f in test_path]
print(len(test_files))
test_images = [img_to_array(load_img(image, target_size = (244, 244,3))) for image in test_files]
#print(test_images[0].shape, len(test_images))
test_arr = np.stack(test_images,axis=0) 
#print(test_arr.shape)
print("Predicting..")
preds = (model.predict(test_arr)).argmax(axis=1)
bar_list = np.where(preds==1)[0]
bars = map(test_files.__getitem__, bar_list)
print("Number of bar charts found:" len(bars))
for i,img in enumerate(bars):
	shutil.copy2(img,'./bars')
	if i%500 == 0:
		print("Moved {0} images".format(i))


