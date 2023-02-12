"""
# File taken from https://github.com/mcordts/cityscapesScripts/
# License File Available at:
# https://github.com/mcordts/cityscapesScripts/blob/master/license.txt

# ----------------------
# The Cityscapes Dataset
# ----------------------
#
#
# License agreement
# -----------------
#
# This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
#
# 1. That the dataset comes "AS IS", without express or implied warranty. Although every effort has been made to ensure accuracy, we (Daimler AG, MPI Informatics, TU Darmstadt) do not accept any responsibility for errors or omissions.
# 2. That you include a reference to the Cityscapes Dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our website; for other media cite our preferred publication as listed on our website or link to the Cityscapes website.
# 3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
# 4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
# 5. That all rights not expressly granted to you are reserved by us (Daimler AG, MPI Informatics, TU Darmstadt).
#
#
# Contact
# -------
#
# Marius Cordts, Mohamed Omran
# www.cityscapes-dataset.net

"""
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!
"""
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  0 ,        0 , 'flat'            , 1       , False        , False        , ( 64,  0,  0) ),
    Label(  'sidewalk'             ,  1 ,        1 , 'flat'            , 1       , False        , False        , (128,128,128) ),
    Label(  'lawn'                 ,  2 ,        2 , 'flat'            , 1       , False        , False        , (128,128,  0) ),
    Label(  'staircase'            ,  3 ,        3 , 'construction'    , 2       , False        , False        , (  0,128,  0) ),
    Label(  'curbs'                ,  4 ,        4 , 'construction'    , 2       , False        , False        , (128,  0,  0) ),
    Label(  'flowerbeds'           ,  5 ,        5 , 'construction'    , 2       , False        , False        , (192,  0,128) ),
    Label(  'chairs'               ,  6 ,        6 , 'construction'    , 2       , False        , False        , ( 64,128,128) ),
    Label(  'walls'                ,  7 ,        7 , 'construction'    , 2       , False        , False        , ( 64,  0,128) ),
    Label(  'posts'                ,  8 ,        8 , 'construction'    , 2       , False        , False        , (128,  0,128) ),
    Label(  'buildings'            ,  9 ,        9 , 'construction'    , 2       , False        , False        , (  0,128,128) ),
    Label(  'trees'                , 10 ,       10 , 'construction'    , 2       , False        , False        , (192,  0,  0) ),
    Label(  'fence'                , 11 ,       11 , 'construction'    , 2       , False        , False        , ( 64,128,  0) ),
    Label(  'manholecover'         , 12 ,       12 , 'construction'    , 2       , False        , False        , (192,128,128) ),
    Label(  'bus'                  , 13 ,       13 , 'vehicle'         , 3       , False        , False        , (  0,  0,128) ),
    Label(  'sky'                  , 14 ,       14 , 'sky'             , 4       , False        , False        , (192,128,  0) ),
    Label(  '_background_'         , 15 ,       15 , 'background'      , 5       , False        , False        , (  0,  0,  0) ),
]
"""
"""
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  0 ,        0 , 'flat'            , 1       , False        , False        , (  0,  0,  0) ),
    Label(  'sidewalk'             ,  1 ,        1 , 'flat'            , 1       , False        , False        , (128,  0,  0) ),
    Label(  'stair'                ,  2 ,        2 , 'flat'            , 1       , False        , False        , (128,192,  0) ),
    Label(  'ramp'                 ,  3 ,        3 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
]
"""
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  0 ,        0 , 'flat'            , 1       , False        , False        , (  0,  0,  0) ),
    Label(  'sidewalk'             ,  1 ,        1 , 'flat'            , 1       , False        , False        , (128,  0,  0) ),
    Label(  'building'             ,  2 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'wall'                 ,  3 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'fence'                ,  4 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'pole'                 ,  5 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'light'                ,  6 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'sign'                 ,  7 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'vegetation'           ,  8 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'terrain'              ,  9 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'sky'                  ,  10 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'pedestrain'           ,  11 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'rider'                ,  12 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'car'                  ,  13 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'truck'                ,  14 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'bus'                  ,  15 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'train'                ,  16 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'motocycle'            ,  17 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'bicycle'              ,  18 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'stair'                ,  19 ,       19 , 'flat'            , 1       , False        , False        , (128,192,  0) ),
    Label(  'curb'                 ,  20 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'ramp'                 ,  21 ,       21 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'runway'               ,  22 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'flowerbed'            ,  23 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'door'                 ,  24 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'CCTV camera'          ,  25 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'Manhole'              ,  26 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'hydrant'              ,  27 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'belt'                 ,  28 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'dustbin'              ,  29 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
    Label(  'ignore'               ,  30 ,      255 , 'background'      , 5       , False        , True         , ( 64,192,128) ),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# label2trainid
label2trainid   = { label.id      : label.trainId for label in labels   }
# trainId to label object
trainId2name   = { label.trainId : label.name for label in labels   }
trainId2color  = { label.trainId : label.color for label in labels  }

color2trainId = { label.color : label.trainId for label in labels   }

trainId2trainId = { label.trainId : label.trainId for label in labels   }

# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print(("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' )))
    print(("    " + ('-' * 98)))
    for label in labels:
        print(("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval )))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print(("ID of label '{name}': {id}".format( name=name, id=id )))

    # Map from ID to label
    category = id2label[id].category
    print(("Category of label with ID '{id}': {category}".format( id=id, category=category )))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print(("Name of label with trainID '{id}': {name}".format( id=trainId, name=name )))
