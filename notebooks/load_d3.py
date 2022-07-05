import h5py
import xarray as xr
import hvplot.xarray

def load_d3(filename, fields=None):
    df = h5py.File(filename,'r')
    tasks = []
    da = []
    names = []
    if fields is None:
        fields = df['tasks'].keys()
    for f in fields:
        data = df['tasks'][f]
        names.append(data.name.split('/')[-1])
        dims = []
        scales = []
        n_tensor = 0
        for i, d in enumerate(data.dims):
            if d.label:
                dims.append(d.label)
            else:
                # hack: assuming if label is missing, it's a coordinate
                dims.append(f'coords_{n_tensor}')
                n_tensor += 1
            try:
                scales.append(d[0][:]) # hack...shouldn't just take first scale
            except RuntimeError:
                scales.append(np.arange(data.shape[i]))
        da.append(xr.DataArray(data[:], coords=dict(zip(dims, scales)),dims=dims))
    df.close()
    return xr.Dataset(dict(zip(names,da)))
        
