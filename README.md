
# Song Match

The audio comparision script uses libraries listed out in the requirements.txt file

## Geting Started

The code is written in python 2.7, and the dependencies can be installed via pip.

**Installation via command line using pip**
> pip install -r requirements.txt


### As library

like in 'Testing.ipynb' file.

**Preprocessing the all_songs folder is a mandate & is a one-time job after refreshing the song list**

`import preprocess
import glob, os
folder = glob.glob(os.getcwd() + '/all_songs/*')
preprocess.store_feat_of_songs(folder)`


**Getting the results**

`import songMatch
import glob, os
featList_folder = glob.glob(os.getcwd() + '/featList/*')
df = songMatch.get_results('test.mp3', featList_folder, min_perc=0.70)
df`

