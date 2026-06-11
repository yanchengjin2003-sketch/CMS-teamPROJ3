1. training args:

```python
learning_rate = 0.0002  # learning rate for the optimizer
batch_size = 12  # batch size for the dataloader
input_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 350  # number of epochs to train the model for
iterations_per_epoch = 64  # number of iterations per epoch
classes = ["endo_lum", "cyto", "endo_mem", "bg"]
force_all_classes = True # only data with all 4 labels can be used for training

# model acrhitecture
model_name = "3d_transunet"  # name of the model to use
model_to_load = "3d_transunet"  # name of the pre-trained model to load
model = TransUNet_3D(1, len(classes), img_size=input_array_info["shape"])
```



2. loss function has been changed:

loss = BCE --> loss =  0.5 * CE+0.5 * DICE



3. Processed and predicted data:

csc predict train_3D.py -c 124,125,131,132,133,135,136,137,138,139,142,143,144,145,150,151,157,171,172,175,183,416,417 -O



There corps are all from the data file "jrc_mus-Liver". Here is the info about this cell: [OpenOrganelle](https://openorganelle.janelia.org/datasets/jrc_mus-liver)