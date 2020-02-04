The following APIs changed from version 0.0.7 to 1.0rc1:

`read(cls, filename, nsideCoverage=None, pixels=None, header=False)` to
`read(cls, filename, nside_coverage=None, pixels=None, header=False)`

`makeEmpty(cls, nsideCoverage, nsideSparse, dtype, primary=None,
sentinel=None)` to `make_empty(cls, nside_coverage, nside_sparse, dtype,
primary=None, sentinel=None)`

`convertHealpixMap(healpixMap, nsideCoverage, nest=True, sentinel=hp.UNSEEN)`
to `convert_healpix_map(healpix_map, nside_coverage, nest=True,
sentinel=hp.UNSEEN)`

`updateValues(self, pixel, values, nest=True)` to `update_values_pix(self,
pixels, values, nest=True)`

`getValueRaDec(self, ra, dec, validMask=False)` and `getValueThetaPhi(self,
theta, phi, validMask=False)` to `get_values_pos(self, theta_or_ra, phi_or_dec,
lonlat=False, valid_mask=False)`

`getValuePixel(self, pixel, nest=True, validMask=False)` to
`get_values_pix(self, pixels, nest=True, valid_mask=False)`

`coverageMap` to `coverage_map`

`coverageMask` to `coverage_mask`

`nsideCoverage` to `nside_coverage`

`nsideSparse` to `nside_sparse`

`isIntegerMap` to `is_integer_map`

`isRecArray` to `is_rec_array`

`generateHealpixMap(self, nside=None, reduction='mean', key=None)` to
`generate_healpix_map(self, nside=None, reduction='mean', key=None)`

`validPixels` to `valid_pixels`

`applyMask(self, maskMap, maskBits=None, inPlace=True)` to `apply_mask(self,
mask_map, mask_bits=None, in_place=True)`

`getSingle(self, key, sentinel=None)` to `get_single(self, key, sentinel=None)`

`makeUniformRandomsFast(sparseMap, nRandom, nsideRandoms=2**23, rng=None)` to
`make_uniform_randoms_fast(sparse_map, n_random, nside_randoms=2**23,
rng=None)`

`makeUniformRandoms(sparseMap, nRandom, rng=None)` to
`make_uniform_randoms(sparse_map, n_random, rng=None)`

`sumUnion(mapList)` to `sum_union(map_list)`

`sumIntersection(mapList)` to `sum_intersection(map_list)`

`productUnion(mapList)` to `product_union(map_list)`

`productIntersection(mapList)` to `product_intersection(map_list)`

`orUnion(mapList)` to `or_union(map_list)`

`orIntersection(mapList)` to `or_intersection(map_list)`

`andUnion(mapList)` to `and_union(map_list)`

`andIntersection(mapList)` to `and_intersection(map_list)`

`xorUnion(mapList)` to `xor_union(map_list)`

`xorIntersection(mapList)` to `xor_intersection(map_list)`

