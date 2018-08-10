import argparse
from train import load_model
from dataset import load_images_features
import os
from dataset import EMOTION2INTEGER
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from matplotlib import pyplot as plt
import itertools

def get_cmd_args():
    """ Parse user command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_dir",default="dataset",type=str)
    parser.add_argument("-w","--weights",type=str,required=True)
    parser.add_argument("-m","--model",type=str,required=True)
    args = parser.parse_args()
    return args
def load_validation_files(dataset_dir):
    test_image_files = []
    test_labels = []
    for emotion_folder in os.listdir(os.path.join(dataset_dir,"test")):
        for img_file in os.listdir(os.path.join(dataset_dir,"test",emotion_folder)):
            test_image_files+=[img_file]
            test_labels+=[EMOTION2INTEGER[emotion_folder]]
    return [test_image_files,test_labels]
def plot_confusion_matrix(cm, classes,
                          title):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    """Start of training program.
    """
    
    args = get_cmd_args()
    model = load_model(args.model,args.weights)
    files,labels = load_validation_files(args.dataset_dir)
    image_shape = model.inputs[0].shape.as_list()[1:]
    IMAGE_HEIGHT = image_shape[0]
    images,dlib_points,dlib_points_distances,dlib_points_angles,labels = load_images_features(os.path.join(args.dataset_dir,"test"),files,labels,image_shape,False)
    y_true = labels

    images = images.astype(np.float32)/255
       
    dlib_points = dlib_points.astype(np.float32)/IMAGE_HEIGHT
    dlib_points_distances = dlib_points_distances.astype(np.float32)/IMAGE_HEIGHT
    dlib_points_angles = dlib_points_angles.astype(np.float32)/np.pi

    dlib_points = dlib_points.reshape(-1,1,68,2)
    dlib_points_distances = dlib_points_distances.reshape(-1,1,68,1)
    dlib_points_angles = dlib_points_angles.reshape(-1,1,68,1)

    y_pred = model.predict([images,dlib_points,dlib_points_distances,dlib_points_angles])

    y_pred = np.argmax(y_pred,axis=1)
    acc = accuracy_score(y_true, y_pred)
    
    cnf_matrix = confusion_matrix(y_true, y_pred)
    class_names = ["anger","disgust","fear","happy","sad","surprise","neutral"]

    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                        title='Emopy Confusion matrix')
    plt.savefig("conf_matrix.png")
    
    plt.show()
    

if __name__ == '__main__':
    main()