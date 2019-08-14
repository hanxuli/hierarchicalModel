from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from scipy import misc
# import numpy
# import numpy.random
# import scipy.ndimage
# import scipy.misc
#
Datagen = ImageDataGenerator(rotation_range=40,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')

img = load_img('../data/jct.png')  # 获取一个PIL图像
x_img = img_to_array(img)
x_img = x_img.reshape((1,) + x_img.shape)

i = 0
for img_batch in Datagen.flow(x_img,
                              batch_size=1,
                              save_to_dir='../data/new',
                              save_prefix='hand',
                              save_format='png'):
    i += 1
    if i > 20:
        break