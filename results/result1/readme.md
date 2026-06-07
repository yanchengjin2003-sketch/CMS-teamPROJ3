# Training process #

```python
learning_rate = 0.0002  # learning rate for the optimizer
batch_size = 5  # batch size for the dataloader
epochs = 100  # number of epochs to train the model for
iterations_per_epoch = 48  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility
classes = ["endo_lum", "cyto", "endo_mem"]

# model acrhitecture
model_name = "3d_transunet"  # name of the model to use
model_to_load = "3d_transunet"  # name of the pre-trained model to load
model = TransUNet_3D(1, len(classes), img_size=input_array_info["shape"])
```



# Processed and predicted data #

csc predict train_3D.py -c 124,125,131,132,133,135,136,137,138,139,142,143,144,145,150,151,157,171,172,175,183,416,417 -O



There corps are all from the data file "jrc_mus-Liver". Here is the info about this cell: [OpenOrganelle](https://openorganelle.janelia.org/datasets/jrc_mus-liver)

Further we will identify volumes that are suitable for training and design heuristic algorithm with prior info of liver cell.

