import cv2
import json
import glob
import os
import keras
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report, accuracy_score
import numpy as np 
import pandas as pd
import shutil
import subprocess
import argparse


def plot_extractor(fold_path):
    dir_name = "./figure-separator/results"
    file_list = glob.glob(dir_name+'/*.json')
    #print(file_list)
    out_path = './figure-separator/extracts/'
    rem_files = glob.glob(out_path+'/*')
    for f in rem_files:
        os.remove(f)
    sub_figs = {}
    for json_file in file_list:
        img_name = json_file.strip('.json').split('/')[-1]
    #print(img_name)  
        image_path = fold_path+'/' +img_name
        #print(image_path)
        try:
            img = cv2.imread(image_path)
        except:
            print(image_path + ' does not exist')
            pass  
        
        with open(json_file) as f:
            sub_figs = json.load(f)
        
        
        #print(sub_figs)
        #print(img)
        
        for i, pos in enumerate(sub_figs):
                y = pos['y']
                x = pos['x']
                conf = pos['conf']
                h = pos['h']
                w = pos['w']
                #print(y,x,conf,h,w)
                save_name = out_path+img_name.strip('.jpg')+'-'+str(i)+'.jpg'
                #print(save_name)
                try:
                    cv2.imwrite(save_name, img[y:y+h,x:x+w])                   

                except:
                    print(save_name + ' is corrupted/does not exist')
                    pass
            
        
    print("All images saved in" + out_path)
    return out_path

def bar_classifier(fold_path, true_bars = None):
    model = load_model('bc_models/classifybars.h5')
    test_path = glob.glob(fold_path+'/*')
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
    not_bar_list = np.where(preds==0)[0]
    bars = map(test_files.__getitem__, bar_list)
    not_bars = map(test_files.__getitem__, not_bar_list)
    #print(bars)
    #print(not_bars)
    print('Number of bar charts found: %d' %len(bars))
    
    if true_bars:
        bdf = pd.DataFrame({'ImageName':bars})
        bdf['Pred Label'] = 'bar'
        nbdf = pd.DataFrame({'ImageName':not_bars})
        nbdf['Pred Label'] = 'not bar'
        pred_bars_df = pd.concat([bdf,nbdf])
        pred_bars_df['ImageName'] = pred_bars_df['ImageName'].str.replace(fold_path,'')
        print(pred_bars_df)
        compute_metrics(pred_bars_df, true_bars)
    
    out_path = './data/plot_rt/plots'
    rem_files = glob.glob(out_path+'/*')
    for f in rem_files:
        os.remove(f)
        
    for i,img in enumerate(bars):
        shutil.copy2(img,out_path)
        if (i%100 == 0 and i != 0) or (i == len(bars)-1):
	        print("Moved {0} images".format(i+1))
	        
    
    test_images = None
    test_array = None
    bar_list = None
    bars = None	
    return out_path
    
def compute_metrics(pred_bars_df,true_bars):
    true_bars_df = pd.read_csv(true_bars) 
    df = pred_bars_df.merge(true_bars_df, on='ImageName')
    #print(df)
    print("Accuracy Score for bar classifier: {}" .format(accuracy_score(df['True Label'], df['Pred Label'])))
    print(classification_report(df['True Label'], df['Pred Label'], labels=['bar','not bar']))
    

def run_figsep(img_dir):
    dir_name = "./figure-separator/results"
    rem_files = glob.glob(dir_name+'/*')
    for f in rem_files:
        os.remove(f)
    
    subprocess.call(["python2", "./figure-separator/main.py", "--images", img_dir, "--annotate", "1"])

def write_image_list(bars_path):
    all_files = os.listdir(bars_path)

    with open('./data/plot_rt/test_real.idl','w') as fh:
        fh.write('\n'.join(r'"plots/'+str(name)+r'";' for name in all_files))
    print('File names written in required format')
    
    return bars_path.strip('plots')


def run_scatteract(image_dir, image_output_dir, csv_output_dir, true_coord_idl=None):
    #print(model_dict, iteration,image_dir, image_output_dir, csv_output_dir , true_coord_idl, predict_idl) 
    #subprocess.call(["python2", "scatter_extract.py", "--model_dict", model_dict, "--iteration", iteration,"--image_dir", image_dir,
#"--predict_idl", predict_idl,"--image_output_dir", image_output_dir,"--csv_output_dir", csv_output_dir,"--true_coord_idl", true_coord_idl])

    
    model_dict = '{"ticks":"./output/ticks_v1", "labels":"./output/labels_v1"}'
    iteration = str(125000)
    predict_idl = './data/plot_rt/test_real.idl'
    if true_coord_idl:
        subprocess.call(["python2", "scatter_extract.py", "--model_dict", model_dict, "--iteration", iteration,"--image_dir", image_dir,
"--predict_idl", predict_idl,"--image_output_dir", image_output_dir,"--csv_output_dir", csv_output_dir,"--true_coord_idl", true_coord_idl])
    else:
        subprocess.call(["python2", "scatter_extract.py", "--model_dict", model_dict, "--iteration", iteration,"--image_dir", image_dir,
"--predict_idl", predict_idl,"--image_output_dir", image_output_dir,"--csv_output_dir", csv_output_dir])



def main():

    parser = argparse.ArgumentParser()

    #parser.add_argument('--model_dict', help='Directory for the object detection models', required=True)
    #parser.add_argument('--iteration', help='Iteration number for the trained models', required=True)
    parser.add_argument('--image_dir', help='Directory of the images', required=True)
    #parser.add_argument('--predict_idl', help='Path of an idl file which list the images to predict on', required=False, default=None)
    parser.add_argument('--image_output_dir', help='Directory to output images with bounding boxes', required=True)
    parser.add_argument('--csv_output_dir', help='Directory to output csv of results', required=True)
    parser.add_argument('--true_labels', help='csv of the ground truth labels', required=False, default=None)
    parser.add_argument('--true_bars', help='csv of the ground truth bars', required=False, default=None)

    args = vars(parser.parse_args())

    run_figsep(args["image_dir"])
    extract_path = plot_extractor(args["image_dir"])
    bars_path = bar_classifier(extract_path, true_bars = args["true_bars"])
    image_dir = write_image_list(bars_path)
    run_scatteract(image_dir, args["image_output_dir"], args["csv_output_dir"],true_coord_idl=args["true_labels"])  
 
	

if __name__ == "__main__":
    main()
            	

