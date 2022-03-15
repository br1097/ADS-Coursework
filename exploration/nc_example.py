import netCDF4 as nc

fn = 'data/hur.T1Hpoint.UMRA2T.19910428_19910501.BOB01.1p5km.nc'
ds = nc.Dataset(fn)
print(ds.__dict__)
