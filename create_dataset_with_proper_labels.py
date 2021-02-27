import os
import json
rootdir = '/home/Daniel/DeepProject/dataset/cut_train_data_360'
#rootdir = '/home/Daniel/DeepProject/dataset/dump_folder'
my_dict = {}
counter = -1
for subdir, dirs, files in os.walk(rootdir):
       # print (os.path.join(subdir, file))
       #if current_speaker !=subdir.split('/')[-1]:
        #current_speaker = subdir.split('/')[-1]
       print(subdir.split('/'))
       if counter>-1:
        my_dict[str(counter)] = subdir.split('/')[-1]
        os.rename(subdir , rootdir+'/'+str(counter))
       counter+=1

with open('/home/Daniel/DeepProject/dataset/speakers_map.json','w') as fp:
    json.dump(my_dict, fp)
