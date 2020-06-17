Requirements:
Python 3 
Dependencies:
numpy, 
keras,
keras-retinanet,
PIL, 
matplotlib,
easygui,
opencv-python,
csv,
seaborn,
tensorflow 

Anaconda distributive recommended since it contains most of packages preinstalled
packages can be installed via command 

pip install packagename

# SEM_analysis
SEM_analysis neural network
Download inference from google drive https://drive.google.com/file/d/1qduzSqZ6V7J-qpSX5C6Ly1OIwW417kt6/view?usp=sharing
Place it to the same folder as inference.py script and run 


python inference.py


For own dataset synthesis:
1. place files with your to textures folder
2. run jupyter_NPS_generate.ipynb for JSON files generation (generated JSON contains particles positions and sizes)
3. run blender_powder.py into blender enviroment
4. It will generate annotations.csv file with all labels and images at directory "renders"

Detailed RetinaNet trainig procedure can be found at https://www.kaggle.com/alexanderkhar/nps-detector
Example of generated dataset can be found at https://www.kaggle.com/alexanderkhar/generated-nps
