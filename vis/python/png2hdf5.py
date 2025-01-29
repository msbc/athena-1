#! /usr/bin/env python

def png2hdf5(png_file, hdf5_file='input.h5', scale=100, invert=True, dataset='data',
             dtype=None, fill_value=-1.0):
    """Convert a PNG file to an HDF5 file."""
    import h5py
    import numpy as np
    from PIL import Image

    dtype = dtype or np.float64

    # Open the PNG file
    with Image.open(png_file) as img:
        img = np.array(img).mean(axis=2)[None, ::-1, :]  # Convert to grayscale

    # Invert the image
    if invert:
        img = 255 - img

    # Scale the image
    if scale:
        img *= scale / 255

    # Fill the image
    img = img.astype(dtype)
    if fill_value is False:
        fill_value = None
    if fill_value is not None:
        img[img == 0] = fill_value

    # Create the HDF5 file
    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset(dataset, data=img)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert a PNG file to an HDF5 file.')
    parser.add_argument('png_file', help='The PNG file to convert.')
    parser.add_argument('hdf5_file', nargs='?', default='input.h5', help='The HDF5 file to create.')
    parser.add_argument('--scale', type=float, default=100, help='The scale factor/max output value.')
    parser.add_argument('--invert', action='store_true', help='Invert the image color (make black max value).')
    parser.add_argument('--dataset', default='data', help='The dataset name in the hdf5 file.')
    parser.add_argument('--dtype', help='The output data type.')
    parser.add_argument('--fill_value', type=float, default=-1.0, help='The fill value.')

    args = parser.parse_args()
    png2hdf5(args.png_file, args.hdf5_file, args.scale, args.invert, args.dataset, args.dtype, args.fill_value)
