# Model Directory

This directory is used to store the model file for the Tumor Segmentation API.

## Required Model File

The API expects the following model file to be present in this directory:

- `best_Attresunet_scripted.pt`: A TorchScript model for brain tumor segmentation

## Setting Up the Model

You can set up the model using the provided `setup_model.py` script:

```
python setup_model.py --model /path/to/your/best_Attresunet_scripted.pt
```

This script will copy the model file to this directory.

Alternatively, you can manually copy the model file to this directory:

```
cp /path/to/your/best_Attresunet_scripted.pt models/
```

## Model Format

The model should be a TorchScript model saved using `torch.jit.script` or `torch.jit.trace`. The model should take a single-channel grayscale image tensor of shape `[1, 1, 256, 256]` as input and output a segmentation mask of the same shape.
