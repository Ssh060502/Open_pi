import h5py

file_path = r"../data/knife.h5"

with h5py.File(file_path, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name} shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"[GROUP]   {name}")
    f.visititems(show)