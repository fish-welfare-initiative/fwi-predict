import ee

from fwi_predict.geo.ee import get_gfs

ee.Initialize()

gfs = get_gfs()
gfs.first()