"""Independent depth corruptions; never use grasp labels or modify RGB."""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, binary_dilation
NEW_KINDS=('irregular_missing','smooth_noise','edge_noise')
LIGHT_VARIANTS={'mild_gaussian':'noise','mild_irregular':'irregular_missing','mild_smooth':'smooth_noise','mild_edge':'edge_noise'}
def depth_corruption(depth,kind,level,rng):
    d=depth.copy();valid=d>0;n=d.shape[0]
    if level not in (1,2,3):raise ValueError('Severity must be 1, 2 or 3')
    if not valid.any():return d
    if kind=='irregular_missing':
        field=gaussian_filter(rng.standard_normal(d.shape),max(1,n/16),mode='reflect')
        fraction=(.1,.3,.5)[level-1]
        mask=field>=np.quantile(field,1-fraction)
        d[mask&valid]=0
    elif kind=='smooth_noise':
        field=gaussian_filter(rng.standard_normal(d.shape),max(1,n/32),mode='reflect')
        field-=field[valid].mean();field/=max(float(field[valid].std()),1e-8)
        cx,cy=rng.uniform(0,n,2);yy,xx=np.indices(d.shape)
        envelope=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*(n/5)**2))
        noise=field*envelope*(.005,.015,.03)[level-1]
        d[valid]=np.maximum(d[valid]+noise[valid],1e-6)
    elif kind=='edge_noise':
        # Depth discontinuities only; ignore missing-data/padding boundaries.
        hi=maximum_filter(np.where(valid,d,-np.inf),size=3)
        lo=minimum_filter(np.where(valid,d,np.inf),size=3)
        neighborhood_valid=minimum_filter(valid.astype(np.uint8),size=3)>0
        edge=valid&neighborhood_valid&((hi-lo)>.02)
        edge=binary_dilation(edge,iterations=level)&valid
        noise=rng.normal(0,(.005,.015,.03)[level-1],int(edge.sum()))
        d[edge]=np.maximum(d[edge]+noise,1e-6)
    else:raise ValueError(kind)
    return d.astype(np.float32)


SEVERITY = {
    "noise": [0, 0.005, 0.015, 0.03],
    "missing": [0, 0.1, 0.3, 0.5],
    "occlusion": [0, 0.1, 0.2, 0.35],
}


def corrupt(rgb, d, kind, level, rng):
    rgb = rgb.copy()
    d = d.copy()
    n = d.shape[0]
    if kind == "noise":
        valid = d > 0
        d[valid] = np.maximum(
            d[valid] + rng.normal(0, SEVERITY[kind][level], valid.sum()), 1e-6
        )
    elif kind == "missing":
        # One contiguous region of missing depth; exact area recorded by protocol.
        side = max(1, round(n * np.sqrt(SEVERITY[kind][level])))
        x, y = rng.integers(0, n - side + 1, 2)
        d[y : y + side, x : x + side] = 0
    elif kind == "occlusion":
        side = max(1, round(n * np.sqrt(SEVERITY[kind][level])))
        x, y = rng.integers(0, n - side + 1, 2)
        rgb[y : y + side, x : x + side] = 0
        d[y : y + side, x : x + side] = 0
    elif kind == "dropout":
        d[:] = 0
    elif kind in NEW_KINDS:
        d = depth_corruption(d, kind, level, rng)
    elif kind != "clean":
        raise ValueError(kind)
    return rgb, d
