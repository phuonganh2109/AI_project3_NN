import math
import os
import numpy as np
from skimage import io
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import gzip
from sklearn import datasets, svm, metrics
import cv2 as cv
import seaborn as sns
from scipy.special import softmax
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import h5py

NUM_HID = 1000
IMG_SIZE = 200
TEST_SIZE = 0.1
RANDOM_STATE = 2020
dict_mnist = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
dict_fas = {0:'T-shirt',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}
dict_dogcat = {0: 'Dog', 1: 'Cat'}

#ghi vao file h5 model train ra duoc
def wr_model(filename, weights):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('weights', data=weights)
    hf.close()

#ghi file csv ket qua predict voi tung anh trong test dataset
def write_csv(data,out,d = False):
    with open('submission.csv','w') as f:
        o = csv.writer(f)
        write_d = []
        if not d:
            o.writerow(['No.','Label'])
            for i in range(len(data)):
                write_d.append([str(data[i]),str(out[i])])
        else:
            o.writerow(['No.','Label','Description'])
            for i in range(len(data)):
               write_d.append([str(data[i]),str(out[i]),d[out[i]]])
        o.writerows(write_d)

def get_images(img_file, number):
    f = open(img_file, "rb") # Open file in binary mode
    f.read(16) # Skip 16 bytes header
    images = []

    for i in range(number):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    return images

def get_labels(label_file, number):
    l = open(label_file, "rb") # Open file in binary mode
    l.read(8) # Skip 8 bytes header
    labels = []
    for i in range(number):
        labels.append(ord(l.read(1)))
    return labels

def convert_png(images, labels, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    for i in range(len(images)):
        out = os.path.join(directory, "%06d-num%d.png"%(i,labels[i]))
        io.imsave(out, np.array(images[i]).reshape(28,28))

def load_mnist(path, kind='train'):
    # Load MNIST data from path
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
        
    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(labels), 28, 28).astype(np.float64)
 
    return images, labels

def input_to_hidden(x,w):
    a = np.dot(x, w)
    a = np.maximum(a, 0, a) # ReLU
    return a

def get_y_train(y):
    y_label = []
    for i in range(len(y)):
        y_train = np.zeros(10)
        y_train[y[i]] = 1
        y_label.append(y_train)
    return y_label

def ELM(img, y):
    out_w, w = 0,0
    w = np.random.normal(size=[img.shape[1], NUM_HID])
    input_X = input_to_hidden(img,w) # input train set for hid layer
    out_w = np.linalg.pinv(input_X) @ y # out of hid layer
    img = input_X@out_w
    return out_w, w

def plot_image_list_count1(label,d):
    labels = []
    #dict_lab = {0:'T-shirt',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}
    for l in label:
        labels.append(d[l])
    plt.figure(figsize = (10,5))
    sns.countplot(labels)
    if d[0]=='T-shirt':
        plt.title('Mnist - fashion')
    else:
        plt.title('Mnist - handwritting')

# show hinh ra, data la t_images or te_images
# labels la t_labels or te_labels
# dict_ la dictionary tuong ung
def show_images1(data, labels, dict_, isTest=False):
    f, ax = plt.subplots(5,5, figsize=(10,10))
    for i,data in enumerate(data[:25]):
        img_num = labels[i]
        img_data = data
        str_label = dict_[img_num]
        if isTest:
            str_label = 'None'
        ax[i//5, i%5].imshow(img_data, cmap = 'gray')
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title("Label: {}".format(str_label),fontsize = 8)
    plt.show()

def show_images(data, isTest=False):
    f, ax = plt.subplots(5,5, figsize=(10,10))
    for i,data in enumerate(data[:25]):
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)
        if label  == 1: 
            str_label='Dog'
        elif label == 0: 
            str_label='Cat'
        if(isTest):
            str_label="None"
        ax[i//5, i%5].imshow(img_data, cmap='gray')
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title("Label: {}".format(str_label),fontsize = 8)
    plt.show()

def mnist():
    t_images, t_labels = load_mnist("./mnist/", kind='train')
    te_images, te_labels = load_mnist("./mnist/", kind='t10k')

    t_img = np.array([i.reshape(28*28) for i in t_images])
    y_train = get_y_train(t_labels)

    te_img = np.array([i.reshape(28*28) for i in te_images])
    y_val = get_y_train(te_labels)

    acc = 0
    acc_arr = []
    we = []
    for i in range(10):
        out_hid, w = ELM(t_img, y_train)
        input_X = input_to_hidden(te_img, w) # input train set for hid layer
        predict = input_X @ out_hid
        tmp = np.argmax(softmax(predict, axis = 1), axis = 1)

        right = [1 if y_val[i][tmp[i]] == 1 else 0 for i in range(len(y_val))]
        if right.count(1) > acc:
            acc = right.count(1)
            we = w
        acc_arr += [right.count(1) / len(y_val)]
    print('Accuracy -', acc / len(y_val))

    #truyen vao tap nhan, t_labels hoac te_labels
    #truyen vao dictionary phu hop voi bo data (dict_fas: fashion, dict_mnist: chu so viet tay)
    #mnist
    no = [i for i in range(len(te_images))]
    write_csv(no,tmp,dict_mnist)
    wr_model('mnist_model.h5',we)
    plot_image_list_count1(t_labels, dict_mnist)
    show_images1(t_images, t_labels, dict_mnist)
    show_images1(te_images, t_labels, dict_mnist, isTest = True)
    
    return acc_arr

def fashion():
    t_images, t_labels = load_mnist("./fashion/", kind='train')
    te_images, te_labels = load_mnist("./fashion/", kind='t10k')

    t_img = np.array([i.reshape(28*28) for i in t_images])
    y_train = get_y_train(t_labels)

    te_img = np.array([i.reshape(28*28) for i in te_images])
    y_val = get_y_train(te_labels)

    acc = 0
    acc_arr = []
    we = []
    for i in range(10):
        out_hid, w = ELM(t_img, y_train)
        input_X = input_to_hidden(te_img, w) # input train set for hid layer
        predict = input_X @ out_hid
        tmp = np.argmax(softmax(predict, axis = 1), axis = 1)


        right = [1 if y_val[i][tmp[i]] else 0 for i in range(len(y_val))]
        if right.count(1) > acc:
            acc = right.count(1)
            we = w

        acc_arr += [right.count(1) / len(y_val)]
    print('Accuracy -', acc / len(y_val))

    #truyen vao tap nhan, t_labels hoac te_labels
    #truyen vao dictionary phu hop voi bo data (dict_fas: fashion, dict_mnist: chu so viet tay)
    #mnist_fashion
    no = [i for i in range(len(te_images))]
    write_csv(no,tmp,dict_fas)
    wr_model('fashion_model.h5',we)
    plot_image_list_count1(t_labels, dict_fas)
    show_images1(t_images, t_labels, dict_fas)
    show_images1(te_images, te_labels, dict_fas, isTest = True)

    return acc_arr

def label_pet_image_one_hot_encoder(img):
    pet = img.split('.')[-3]
    if pet == 'cat': return [1,0]
    elif pet == 'dog': return [0,1]

def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(DATA_FOLDER,img)
        if(isTrain):
            label = label_pet_image_one_hot_encoder(img)
        else:
            label = img.split('.')[0]
        img = cv.imread(path,cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (IMG_SIZE,IMG_SIZE))
        data_df.append([np.array(img),np.array(label)])
    return data_df

def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split('.')[-3])
    sns.countplot(labels)
    plt.title('Cats and Dogs')

def dogcat():
    PATH = './'
    train_image_path = os.path.join(PATH, "train.zip")
    test_image_path = os.path.join(PATH, "test1.zip")

    import zipfile
    with zipfile.ZipFile('./dogs-vs-cats.zip',"r") as z:
        z.extractall(".")
    with zipfile.ZipFile(train_image_path,"r") as z:
        z.extractall(".")
    with zipfile.ZipFile(test_image_path,"r") as z:
        z.extractall(".")

    TRAIN_FOLDER = './train/'
    TEST_FOLDER =  './test1/'

    train_image_list = os.listdir(TRAIN_FOLDER)
    test_image_list = os.listdir(TEST_FOLDER)

    plot_image_list_count(train_image_list)
    plot_image_list_count(os.listdir(TRAIN_FOLDER))

    train = process_data(train_image_list, TRAIN_FOLDER)
    show_images(train)

    test = process_data(test_image_list, TEST_FOLDER, False)
    show_images(test,True)

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE*IMG_SIZE)
    y = np.array([i[1] for i in train])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
# initial
    acc = 0
    w = None
    acc_arr = []
    out = []
    for _ in range(10):
        weights = np.random.normal(size=[X_train.shape[1], NUM_HID])

        input_X = input_to_hidden(X_train, weights)
        out_hid = np.linalg.pinv(input_X) @ y_train

        input_X_val = input_to_hidden(X_val, weights)
        predict = input_X_val @ out_hid
        
        tmp = np.argmax(softmax(predict, axis = 1), axis = 1)
        right = [1 if y_val[i][tmp[i]] else 0 for i in range(len(y_val))]
        if right.count(1) >  acc:
            acc = right.count(1)
            out = out_hid
            w = weights
        acc_arr += [right.count(1) / len(y_val)]
            
    print('Accuracy -', acc / len(y_val))

    X_te = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE*IMG_SIZE)
    input_X_te = input_to_hidden(X_te, w)
    predict = input_X_te @ out
    tmp = np.argmax(softmax(predict, axis = 1), axis = 1)
    write_csv(test_image_list,tmp)
    wr_model('dogs_cats_model.h5',w)

    return acc_arr

def draw(acc):
    x = [int(i + 1) for i in range(len(acc))]
    y1 = acc
   
    y2 = [(1 - a) for a in acc]
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    plt.plot(x, y1, label = 'Accuracy')

    plt.xlabel('Loop')
    plt.ylabel('Accuracy')

    plt.title('ACCURACY')
    plt.legend()
    plt.show()


def menu():
    print('1. Dogs, Cat Recognition.')
    print('2. Clothing Recognition.')
    print('3. Hand Written Recognition.')
    opt = int(input('Input your option:'))
    if opt == 1:
        draw(dogcat())
    if opt == 2:
        draw(fashion())
    if opt == 3:
        draw(mnist())


menu()