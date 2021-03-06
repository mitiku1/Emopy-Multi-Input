from __future__ import print_function
from sklearn.model_selection import train_test_split
import argparse
import os
import pandas as pd
import numpy as np
import cv2
from utils import EMOTIONS
import random, string

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_dir",required=True,help="fer2013.csv file parent directory")
    parser.add_argument("-o","--output_dir",required=True,help="output directory to save images from csv file")

    args = parser.parse_args()
    return args
def generate_random_string(length):
    output = ""
    for i in range(length):
        output+=random.choice(string.ascii_letters+string.digits)
    return output
def preprocess_fer2013(args):
    """This method reads  fer2013 dataset images from csv file and splits into train and test folders.
    
    Arguments:
        args {dict} -- Dictionary which contains dataset_dir and output_dir
    
    Raises:
        Exception -- If a given dataset dir does not exist or fer2013.csv file does not exist inside given directory.
    """

    if not os.path.exists(args.dataset_dir):
        raise Exception("Dataset dir "+args.dataset_dir+" doesnot exist")
    elif not os.path.exists(os.path.join(args.dataset_dir,"fer2013.csv")):
        raise Exception("Directory "+args.dataset_dir+" doesnot contain fer2013.csv")
    else:
        
        df = pd.read_csv(os.path.join(args.dataset_dir,"fer2013.csv"),sep=",",names=["emotion","pixels","Usage"],header=0)
       
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        if not os.path.exists(os.path.join(args.output_dir,"train")):
            os.mkdir(os.path.join(args.output_dir,"train"))
        if not os.path.exists(os.path.join(args.output_dir,"test")):
            os.mkdir(os.path.join(args.output_dir,"test"))

        for em in EMOTIONS.keys():
            if not os.path.exists(os.path.join(args.output_dir,"train",EMOTIONS[em])):
                os.mkdir(os.path.join(args.output_dir,"train",EMOTIONS[em]))

        for em in EMOTIONS.keys():
            if not os.path.exists(os.path.join(args.output_dir,"test",EMOTIONS[em])):
                os.mkdir(os.path.join(args.output_dir,"test",EMOTIONS[em]))
        
        for index,row in df.iterrows():

            emotion = EMOTIONS[row["emotion"]]
            pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=' ')
            pixels = pixels.reshape(48,48)

            if row["Usage"] == "Training":
                cv2.imwrite(os.path.join(args.output_dir,"train",emotion,generate_random_string(10)+".png"),pixels)
          
            elif row["Usage"] == "PrivateTest" or row["Usage"]=="PublicTest":
                cv2.imwrite(os.path.join(args.output_dir,"test",emotion,generate_random_string(10)+".png"),pixels)
  
            
            if (index+1) %1000 ==0:
                print ("processed ",(index+1),"images")
        

def main():
    args = get_args()
    preprocess_fer2013(args)

if __name__ == '__main__':
    main()