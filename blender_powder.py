# Blender script for rendering synthetic powder images
# Brian DeCost -- Carnegie Mellon University -- 2016
import bpy

import os
import sys
import json
import numpy as np
import csv
import time

# Blender python scripts can't take command-line arguments directly
# one solution: pass arguments by setting shell environment variables.
#
# TEXTUREPATH: path to particle surface texture image source
# PARTICLESPATH: path particle sizes and positions (json data)
# RENDERPATH: path to write rendered image

# try to pull arguments from environment variables
# if not set: go with a reasonable default
#params=open('params').read()
#params=params.split(' ')
#print(os.listdir())
#bpy.ops.wm.read_factory_settings()

# GET BBOXES

def camera_view_bounds_2d(scene, camera_object, mesh_object):
    """
    Returns camera space bounding box of the mesh object.

    Gets the camera frame bounding box, which by default is returned without any transformations applied.
    Create a new mesh object based on mesh_object and undo any transformations so that it is in the same space as the
    camera frame. Find the min/max vertex coordinates of the mesh visible in the frame, or None if the mesh is not in view.

    :param scene:
    :param camera_object:
    :param mesh_object:
    :return:
    """

    """ Get the inverse transformation matrix. """
    matrix = camera_object.matrix_world.normalized().inverted()
    """ Create a new mesh data block, using the inverse transform matrix to undo any transformations. """
    mesh = mesh_object.to_mesh()#scene, True, 'RENDER')
    mesh.transform(mesh_object.matrix_world)
    mesh.transform(matrix)

    """ Get the world coordinates for the camera frame bounding box, before any transformations. """
    frame = [-v for v in camera_object.data.view_frame(scene=scene)[:3]]

    lx = []
    ly = []

    for v in mesh.vertices:
        co_local = v.co
        z = -co_local.z

        if z <= 0.0:
            """ Vertex is behind the camera; ignore it. """
            continue
        else:
            """ Perspective division """
            frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)
    try:
        bpy.data.meshes.remove(mesh)
    except: None
    """ Image is not in view if all the mesh verts were ignored """
    if not lx or not ly:
        return None

    min_x = np.clip(min(lx), 0.0, 1.0)
    min_y = np.clip(min(ly), 0.0, 1.0)
    max_x = np.clip(max(lx), 0.0, 1.0)
    max_y = np.clip(max(ly), 0.0, 1.0)

    """ Image is not in view if both bounding points exist on the same side """
    if min_x == max_x or min_y == max_y:
        return None

    """ Figure out the rendered image size """
    render = scene.render
    fac = render.resolution_percentage * 0.01
    dim_x = render.resolution_x * fac
    dim_y = render.resolution_y * fac

    return (min_x, min_y), (max_x, max_y)




def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles



PATH='C:\\Users\\User\\Downloads\\DeCost-Holm_Data-in-Brief\DeCost-Holm_Data-in-Brief'
#texture_path=PATH+'\\'+'part_texture.jpg'#params[0]
particles_path=PATH+'\\'+'particles/'#params[1]
renderpath=PATH+'\\''renders/'#params[2]
#texture_pathNPs=PATH+'\\'+'metal_texture.jpg'
texturespath=PATH+'\\'+'textures'+'\\'+'newtext'#params[0]



jsonfiles=[]
for jsonfile in os.listdir(particles_path):
    if jsonfile[-5:]=='.json':
        jsonfiles.append(particles_path+'\\'+ jsonfile)
# delete the cube that's loaded by default
# it should already be selected when blender starts

# set up a few more lamps
def newlamp(name, lamptype, loc):
    bpy.ops.object.add(type='LIGHT', location=loc)
    lamp = bpy.context.object
    lamp.name = str(name)
    lamp.data.name = 'Light{}'.format(name)
    lamp.data.type = lamptype
    return lamp

def newlamploc():
    xy=(np.random.random(2)-0.5)*10
    zed=np.random.random()*20.0
    return (xy[0],xy[1],zed)

for o in bpy.context.scene.objects:
    if o.type == 'MESH':
        o.select_set(True)
    else:
        o.select_set(False)

bpy.ops.object.delete()
        
scene = bpy.data.scenes["Scene"]
# Set render resolution
scene.render.resolution_x = 512
scene.render.resolution_y = 512
# Set camera rotation in euler angles
scene.camera.rotation_mode = 'XYZ'
scene.camera.rotation_euler = (0.0, 0.0, 0.0)
# set the camera position
scene.camera.location.x = 0
scene.camera.location.y = 0
scene.camera.location.z = 10

# first choose the cycles rendering engine
bpy.context.scene.render.engine = 'CYCLES'

# set up material for the particles
texturesfullpathes=[]

for f in os.listdir(texturespath):
    if os.path.isfile(texturespath+'\\'+f):
        texturesfullpathes.append(texturespath+'\\'+ f)

texturepathesall=getListOfFiles(texturespath)
              
with open('C:\\Users\\User\\Downloads\\DeCost-Holm_Data-in-Brief\DeCost-Holm_Data-in-Brief\\annotations2.csv', 'a',newline='') as csvfile:
    filewriter = csv.writer(csvfile)
    timebeg=time.time()
    for ind,jsonfile in enumerate(jsonfiles):
        #[f for f in os.listdir('.') if os.path.isfile(f)]
        if os.path.isfile(renderpath+'\\'+jsonfile.split('\\')[-1][:-5]+'.png'):
            print ('PASS ', jsonfile)
            continue
        timeit=time.time()
        for o in bpy.context.scene.objects:
            if o.type in ['MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'ARMATURE', 'LATTICE', 'EMPTY', 'LIGHT', 'SPEAKER']:
                o.select_set(True)
            else:
                o.select_set(False)
        bpy.ops.object.delete()
        #print (jsonfile)
        texture_path=np.random.choice(texturesfullpathes)
        texture_pathNPs=np.random.choice(texturesfullpathes)
        #print(texture_path, texture_pathNPs)
        locs=[(0,0,10),newlamploc(),newlamploc(),newlamploc()]   
        lights=[]
        #print (locs)
        for i, loc in enumerate(locs):
            if i in [0,1,2,3]:
                lights.append(newlamp(i, 'SUN', loc))
        lamp_objects = [o for o in bpy.context.scene.objects if o.type=='LIGHT']
        # SET BRIGHTNES
        for o in lamp_objects:
            o.data.use_nodes = True
            o.data.node_tree.nodes['Emission'].inputs['Strength'].default_value = np.random.rand()*1.0
        #print(os.listdir(texturespath))
        #LOAD TEXTURES FROM IMAGES
        mat = bpy.data.materials.new('thematerial')
        mat.diffuse_color = (1,1,1,1)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = bpy.data.images.load(texture_path)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        matNPs = bpy.data.materials.new('thematNPserial')
        matNPs.diffuse_color = (1,1,1,1)
        matNPs.use_nodes = True
        bsdfN = matNPs.node_tree.nodes["Principled BSDF"]
        texImage = matNPs.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = bpy.data.images.load(texture_pathNPs)
        matNPs.node_tree.links.new(bsdfN.inputs['Base Color'], texImage.outputs['Color'])
        bpy.ops.mesh.primitive_plane_add(size=15, location=(0.0, 0.0, -2.0),rotation=(0.0, 0.0, np.pi * 2 * np.random.random()))
        bpy.ops.object.mode_set(mode='EDIT')
        #bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.object.data.materials.append(mat)  
        bpy.ops.object.shade_smooth()
            
        # load particle dataset    
        with open(jsonfile, 'r') as f:
            dataset = json.load(f)
        '''
        print('distribution: {}'.format(dataset['distribution']))
        print('mean: {}'.format(dataset['loc']))
        print('sigma: {}'.format(dataset['shape']))
        print('timestamp: {}'.format(dataset['timestamp']))
        '''
        # particle positions are specified on [0,1]
        # set the size of the render box:
        scale = np.array([11.0, 11.0, 2.0])

        # build spherical mesh model for each particle
        #print ('startParticles')
        particles=[]
        for num, particle in enumerate(dataset['particles'][:]):
            #print (num)
            size = particle['size']
            
            # particle positions are specified on [0,1] -- scale this to fill the render volume
            loc = np.array([particle['x'], particle['y'], particle['z']])
            x, y, z = scale * loc - scale/2
            z=size-2
            # choose random Euler angles to rotate each particle
            a1, a2, a3 = np.pi / 2 * np.random.random(3)

            # create spherical mesh for each particle
            bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=size/2, location=(x,y,z), rotation=(a1,a2,a3))

            # set up for texture unwrapping
            bpy.ops.object.mode_set(mode='EDIT')
            #bpy.ops.uv.smart_project()
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.object.data.materials.append(matNPs)      
            bpy.ops.object.shade_smooth()
            particles.append(bpy.context.selected_objects[-1])
        # switch render engine back
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        xyrend=[]
        for particle in particles:
            xyrend.append(camera_view_bounds_2d (scene, scene.camera, particle))
        #print (xyrend)
        # render the scene and save to RENDERPATH
        #bpy.ops.render.shutter_curve_preset(shape='SMOOTH')
        scene.render.filepath = renderpath+'\\'+jsonfile.split('\\')[-1][:-5]+'.png'
        bpy.ops.render.render(write_still=True)
        timeitfin=time.time()
        print (ind, ' of ', len(jsonfiles), ' iteration time is ', timeitfin-timeit, 'alltime is', timeitfin-timebeg)
        for xysez in xyrend:
             if xysez!=None:
                coords=int(512*xysez[0][0]), 512-int(512*xysez[1][1]), int(512*xysez[1][0]), 512-int(512*xysez[0][1])
                if coords[2]-coords[0]>0 and coords[3]-coords[1]>0:
                    filewriter.writerow([scene.render.filepath, coords[0],coords[1],coords[2],coords[3], 1])
        if ind==3000: break