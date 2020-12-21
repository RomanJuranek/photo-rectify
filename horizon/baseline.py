from image_geometry.utils import *
from scipy.ndimage import gaussian_filter1d

##
def create_samples_from_groups(groups, picks):

    pts = groups.max()
    si = []
    for i in range(pts+1):
        ind = np.nonzero(groups == i)[0]
        if ind.size == 2:
            si.append(np.array([ind[0], ind[1]], ndmin = 2))
        else:
            choice = np.random.choice(ind, (ind.size * ind.size // 4, 2))
            not_same = choice[:,0]!=choice[:,1]
            si.append(choice[not_same,:])

    if not si:
        s = []
        return s

    s = np.concatenate(si)

    if s.size > picks:
        s = s[np.random.choice(s.shape[0], picks)]

    return s
##
def accumulate_on_zenith(zenith_line, lines, s, pp, prior = None):

    P = np.cross(lines.homogenous()[s[:, 0]], lines.homogenous()[s[:, 1]])
    Pnz = np.abs(P[:, 2]) > 1e-4
    P2 = P[Pnz, 0:2] / P[Pnz, 2].reshape(-1, 1)
    d = P2 @ np.array([-zenith_line[1], zenith_line[0]]).reshape(2, 1)

    dir = np.abs(np.sum(lines.normalized_normal_vector()[s[:, 0]] * lines.normalized_direction_vector()[s[:, 1]],axis=1))
    w = dir[Pnz,None]

    bin_px = 2
    min_d, max_d = np.percentile(d, [5, 95])
    min_d = np.min([min_d,0])
    max_d = np.max([max_d,pp[0]*2])
    diff_d = max_d - min_d
    var_px = pp[1]*2/100

    bin_count = (diff_d / bin_px).astype('int')

    count, bins = np.histogram(d, np.linspace(min_d, max_d, bin_count),weights = w)
    g = gaussian_filter1d(count.astype("f"), var_px)

    try:
        max_bin = np.argmax(g)
        x = (bins[max_bin] + bins[max_bin + 1]) / 2
    except:
        x = pp[0]/2*3

    return x
##
def detect_horizon(image_dict, **kwargs):

    lines = image_dict['lines']
    pp = image_dict['pp']
    groups = image_dict['groups'].astype('i')
    img_h, img_w = image_dict['shape']

    picks = kwargs["picks"]

    #find vps for groups and get zenith
    zenith_line = np.array([1, 0, 0])
    zenith_point = np.array([0, 1, 0])

    v_group = -2
    min_dot = 1
    scale = max([img_h, img_w])
    shift = pp
    num_g = np.min([4, groups.max()+1])
    for i in range(num_g):
        vp = fit_vanishing_point(lines.gather(groups == i).scale_homogenous(scale, shift))
        vp[0:2] = vp[0:2]*scale + shift

        dir = vp[0:2]-pp
        dir = dir/np.linalg.norm(dir)
        dot = np.abs(np.dot(dir, np.array([1,0])))
        if dot < min_dot and dot < 0.5 and np.abs(vp[1] - pp[1]) > np.abs(pp[1]):
            v_group = i
            zenith_point = vp
            zenith_line = np.array([-dir[1],dir[0], dir[1]*pp[0]-dir[0]*pp[1]])
            min_dot = dot.copy()
            break

    if zenith_line[2] > 0:
        zenith_line *= -1

    #remove lines passing through zenith
    groups_h = groups[groups != v_group]
    l_horizontal = lines.gather(groups != v_group)

    #create pairs
    s = create_samples_from_groups(groups_h, picks)

    if len(s) == 0:
        x = pp[1]
    else:
        # accumulate on zenith line
        x = accumulate_on_zenith(zenith_line, l_horizontal, s, pp, None)

    #calculate horizon line and image intersections
    horizon_line = np.array([-zenith_line[1], zenith_line[0], -x])

    return zenith_point, horizon_line