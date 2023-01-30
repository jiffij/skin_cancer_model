import pandas as pd
import os
import shutil

data_dir = os.getcwd()+"/../input/HAM10000_raw"
dest_dir = os.getcwd() + "/../input/reorganized/"

skin_df = pd.read_csv('../input/HAM10000_metadata.csv')
print(skin_df['dx'].value_counts())

label=skin_df['dx'].unique().tolist()  #Extract labels into a list
label_images = []


# Copy images to new folders
for i in label:
    os.mkdir(dest_dir + str(i) + "/")
    sample = skin_df[skin_df['dx'] == i]['image_id']
    label_images.extend(sample)
    for id in label_images:
        shutil.copyfile((data_dir + "/"+ id +".jpg"), (dest_dir + i + "/"+id+".jpg"))
    label_images=[]    