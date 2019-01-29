import h5py
import numpy as np

def save_dict_to_file(hdf5file, d, path = '/'):

    for key, item in d.items():
        if isinstance(item, dict):
            save_dict_to_file(hdf5file, item, path + key + '/')
        elif isinstance(item, list):
            try:
                cnv_np = np.asarray(item)
                if cnv_np.dtype == object:
                    raise ValueError('Did not convert properly to a numpy ndarray')
                hdf5file[path + key] = cnv_np #.astype('|S9')
            except:
                ln = len(item)
                for i, inner_item in enumerate(item):
                # if isinstance(inner_item, dict):
                    key_new = key_with_zeros(i, ln)
                    save_dict_to_file(
                        hdf5file,
                        {key_new: inner_item},
                        path + key + '/'
                    )
        elif isinstance(item, np.ndarray):
            hdf5file[path + key] = item #.astype('|S9')
        elif isinstance(item, (np.int64, np.float64, np.float, float, np.float32, int)):
            hdf5file[path + key] = item
        elif isinstance(item, (str, np.bytes_)):
            hdf5file[path + key] = np.string_(item)


def key_with_zeros(indice, size):
    if size > 1:
        zeros = int(np.log10(size - 1))
    else:
        zeros = 0
    n = zeros*'0' + str(indice)
    n = n[-(zeros+1):]
    return n
