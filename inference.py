import numpy as np
import pandas as pd
import matplotlib
import matplotlib.image as img
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image, ImageFont, ImageDraw, ImageEnhance,ImageOps
import keras_retinanet
from keras_retinanet import preprocessing
from keras_retinanet.preprocessing import csv_generator
from keras_retinanet import models as mdls
from keras_retinanet.bin import convert_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import cv2
from matplotlib.widgets import Slider
import easygui
from matplotlib.widgets import Button
import csv
import seaborn as sns
import tensorflow
import tensorflow.keras as keras

def read_image_bgr(path):
    """ Read an image in BGR format.
    Args
        path: Path to the image.
    """
    # We deliberately don't use cv2.imread here, since it gives no feedback on errors while reading the image.
    #image=Image.open(path)
    image = np.asarray(Image.open(path))
    image2=np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    if image.dtype=='uint16':
        #print('DIVIDE256')
        image=image//256
    image2=np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    try:
        image2[:,:,:]=image
    except:
        image2[:,:,0]=image
        image2[:,:,1]=image
        image2[:,:,2]=image
    return image2[:, :, ::-1].copy()
	
def eqhistrgb(img):
    R, G, B = cv2.split(img)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    #(output1_R,output1_G,output1_B) = (R, G, B)
    return cv2.merge((output1_R, output1_G, output1_B))

def predictonimageparts(image,model, numberofwindows):
    allboxes=[]
    subimages=[]
    alllabels=[]
    allscores=[]
    pixelshift=image.shape[0]//numberofwindows
    for i in range(numberofwindows**2):
        img=image[i%numberofwindows*pixelshift:i%numberofwindows*pixelshift+pixelshift,i//numberofwindows*pixelshift:i//numberofwindows*pixelshift+pixelshift,:].copy()
        subimages.append(img)
        img=preprocess_image(img)
        img, scale = resize_image(img)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(img, axis=0))
        boxes /= scale
        boxes=boxes+np.array([i//numberofwindows*pixelshift,i%numberofwindows*pixelshift,i//numberofwindows*pixelshift,i%numberofwindows*pixelshift])
        allboxes.append(boxes)
        allscores.append(scores)
        alllabels.append(labels)
    allboxes=np.concatenate(allboxes,axis=1)
    allscores=np.concatenate(allscores,axis=1)
    alllabels=np.concatenate(alllabels,axis=1)
    #indiciessorted=np.sort
    return allboxes,allscores,alllabels

def getscale(filename):
    try:
        hdl=open(filename,'r')
        for line in hdl:
            if line.startswith("PixelSizeX="):
                ans=line.split('=')[-1][:-1].split('e')
                return float(ans[0])*10**(float(ans[1])+9)
    except:
        return (0)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

def askforscale():
    scale=easygui.enterbox(msg='Enter number of nm per pixel manually', title='HDR file not found!', default=1, strip=True, image=None, root=None)
    try: scale=float(scale)
    except: scale=10.0
    #print (scale)
    return scale


def opentescanfile(filepath,model, sizesens,neg=False, eqhist=True):
    FACTOR=2
    sizesens=int(sizesens)
    labels_to_names = {0: 'NP'}
    imager = read_image_bgr(filepath)
    width = int(imager.shape[1] /FACTOR)
    height = int(imager.shape[0]/FACTOR)
    dim=(width, height)
    imager = cv2.resize(imager, dim, interpolation = cv2.INTER_AREA)
    #imager.resize(512,512)
    imagertoret=imager.copy()
    hdrfile=filepath[:-4]+'-tif.hdr'
    HDRfile=filepath[:-4]+'-tif.hdr'
    scale=getscale(HDRfile)
    if eqhist==True:
        imager=eqhistrgb(imager)
    imager = cv2.blur(imager,(3,3))
    if neg==True:
        imager=255-imager
    imager = cv2.blur(imager,(3,3))
    if scale==0:
        scale=askforscale()
    print('scale set as ', scale ,' nm per pixel')
    image=imager[:min(imager.shape[0],imager.shape[1]),:min(imager.shape[0],imager.shape[1]),:]
    # process image
    #start = time.time()
    boxes, scores, labels = predictonimageparts(image,model,sizesens)
    BSL=[(boxes[0][i],scores[0][i],labels[0][i]) for i in range(len(boxes[0]))]
    #print (boxes.shape)
    #BSL2=BSL.copy()
    remindex=[]
    for num1, box1 in enumerate(BSL):
        for num2, box2 in enumerate(BSL):
            if num1!=num2:
                if bb_intersection_over_union(box1[0], box2[0])>0.3 and box1[1]>box2[1]:
                    remindex.append(num2)
    listofinds=[i for i in range(len(boxes[0])) if i not in remindex]
    boxes,scores,labels=boxes[0][listofinds],scores[0][listofinds],labels[0][listofinds]
    scale=scale*FACTOR
    return imagertoret, boxes, scores,labels, scale
def update(val):
    global validboxes
    TRASHOLD=probabilitySlider.val
    MAXSIZE=maxsizeSlider.val
    draw = imager.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    validboxes=drawvalidboxes(boxes, scores, labels, TRASHOLD,draw,MAXSIZE)
    ax_orig.imshow(draw)
    fig.canvas.draw()
    #fig.canvas.draw_idle()
    ax_neu.clear()
    sizes=[]
    for box in validboxes:
        sizes.append(np.abs(box[2]-box[0])*SCALE)
    sizes = pd.Series(sizes)   
    #sizes.hist(bins=30, ax=ax_neu)
    textstr='Mean is {} nm +- {} nm \n Total {} particles'.format(str(np.mean(sizes))[:4], str(np.std(sizes))[:4], str(len(sizes)))
    sns.distplot(sizes, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=ax_neu).set_title(textstr)
    ax_neu.set(xlabel='size, nm', ylabel='normalized number of species')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #ax_neu.text(0.5, 0.5, textstr, horizontalalignment='center', verticalalignment='center')#, transform=ax.transAxes)
    #ax_neu.text(textstr, fontsize=14, verticalalignment='top',bbox=props)
    #ax_neu.text=textstr

def boxselector(boxes,scores, labels):
    BSL=[(boxes[i],scores[i],labels[i]) for i in range(len(boxes))]
    remainindex=[]
    for num1, box1 in enumerate(BSL):
        for num2, box2 in enumerate(BSL):
            if num1!=num2:
                if bb_intersection_over_union(box1[0], box2[0])>0.7 and box1[1]>box2[1]:
                    remainindex.append(num1)
    listofinds=remainindex
    boxes,scores,labels=boxes[listofinds],scores[listofinds],labels[listofinds]
    return boxes, scores, labels

def save_dialog(event):
    sizes=[]
    tosave=[]
    savepath=easygui.filesavebox(msg='select filename', title='save', default='test', filetypes='csv')
    if savepath[-4:]!='.csv':
        savepath=savepath+'.csv'
    for box in boxes:
        #if score > TRASHOLD:
        sizes.append(np.abs(box[2]-box[0])*SCALE)
        tosave.append(np.array((box[0],box[1],box[2],box[3])))
    sizes = np.array(sizes)
    tosave=np.array(tosave)
    np.savetxt(savepath, sizes, delimiter=",",header='diameter')
    np.savetxt(savepath+'_boxes.csv', tosave, delimiter=",")
    #print (sizes)
    #pd.DataFrame(np_array).to_csv(savepath)
    return 0

def askforsensitivity():
    msg ="select size sensitivity (2 or 3 for small NPs)"
    title = "size sensitivity"
    choices = [1,2,3,4,5]
    choice = easygui.choicebox(msg, title, choices)
    return choice

def drawvalidboxes(boxes, scores, labels, TRASHOLD, draw,MAXSIZE=10000000000):
    validboxes=[]
    for box, score, label in zip(boxes, scores, labels):
        #DRAW VALID BOXES
        if score > TRASHOLD and ((box[2]-box[0])*SCALE<MAXSIZE and (box[3]-box[1])*SCALE<MAXSIZE):
            validboxes.append(box)
            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color,thickness=1)
    return validboxes


TRASHOLD=0.5
validboxes=[]
model=mdls.load_model('infermodelEQOnly_BS4.05-1.1397-1.3918h5')

#SELECT FILE
def newfile_dialog(event,fig):
	filepath = easygui.fileopenbox()

	#SELECT GRID
	choice=askforsensitivity()
	#GET ALL BOXES
	global imager
	global boxes
	global scores
	global labels
	global SCALE
	global draw
	imager, boxes, scores,labels, SCALE = opentescanfile(filepath,model,choice)

	#PLOT IMAGE
	draw = imager.copy()
	draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
	#SELECT AND DRAW VALID BOXES
	color = label_color(0)
	validoboxes=drawvalidboxes(boxes, scores, labels, TRASHOLD,draw)
	ax_orig.imshow(draw)
	fig.canvas.draw()
	plt.show()
	return 0

filepath = easygui.fileopenbox()

#SELECT GRID
choice=askforsensitivity()
#GET ALL BOXES
imager, boxes, scores,labels, SCALE = opentescanfile(filepath,model,choice)

#PLOT IMAGE
fig = plt.figure(figsize=(6, 4))
ax_orig = fig.add_subplot(121) 
ax_neu = fig.add_subplot(122) 
plt.axis('off')
draw = imager.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
#SELECT AND DRAW VALID BOXES
color = label_color(0)
validoboxes=drawvalidboxes(boxes, scores, labels, TRASHOLD,draw)
ax_orig.imshow(draw)
fig.canvas.draw()

probabilityax  = fig.add_axes([0.13, 0.03, 0.34, 0.02])
probabilitySlider = Slider(probabilityax, 'BBOX TRASHOLD', 0, 1, valinit=0.5)
probabilitySlider.on_changed(update)

maxsizeax = fig.add_axes([0.13, 0.01, 0.34, 0.02])
maxsizeSlider = Slider(maxsizeax, 'MaxSize', 0, SCALE*imager.shape[0], valinit=SCALE*imager.shape[0]/2)
maxsizeSlider.on_changed(update)

axcut = plt.axes([0.88, 0.02, 0.1, 0.04])
bcut = Button(axcut, 'save_hist', color='violet', hovercolor='green')
bcut.on_clicked(save_dialog)
axcut2 = plt.axes([0.76, 0.02, 0.1, 0.04])
bcut2 = Button(axcut2, 'open new file', color='violet', hovercolor='green')
bcut2.on_clicked(lambda x: newfile_dialog(x,fig))
plt.show()

#plt.connect('button_press_event', on_click)

#plt.connect('button_press_event', on_click)
