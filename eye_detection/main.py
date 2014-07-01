import numpy as np 
import cv2 as cv
from eye_test import *
from _ast import TryExcept


MIN_RATIO = 0.3
PUPIL_MIN = 40.0
PUPIL_MAX = 150.0
INITIAL_ELLIPSE_FIT_THRESHOLD = 1.8
STORNG_PERIMETER_RATIO_RANGE = .8, 1.1
STRONG_AREA_RATIO_RANGE = .6, 1.1
FINAL_PERIMETER_RATIO_RANGE = .6, 1.2
STRONG_PRIOR = None

COLORED_IMAGE = getROI(cv.imread("0-eye.png"))
IMG = cv.cvtColor(COLORED_IMAGE, cv.COLOR_BGR2GRAY)	
COARSE_PUPIL_WIDTH = IMG.shape[0] / 2
PADDING = COARSE_PUPIL_WIDTH / 4


def filter_subsets(l):
    return [m for i, m in enumerate(l) if not any(set(m).issubset(set(n)) for n in (l[:i] + l[i + 1:]))]


def ellipse_support_ratio(e, contours):
	a, b = e[1][0] / 2., e[1][1] / 2.  # major minor radii of candidate ellipse
	ellipse_area = np.pi * a * b
	ellipse_circumference = np.pi * abs(3 * (a + b) - np.sqrt(10 * a * b + 3 * (a ** 2 + b ** 2)))
	actual_area = cv.contourArea(cv.convexHull(np.concatenate(contours)))
	actual_contour_length = sum([cv.arcLength(c, closed=False) for c in contours])
	area_ratio = actual_area / ellipse_area
	perimeter_ratio = actual_contour_length / ellipse_circumference  # we assume here that the contour lies close to the ellipse boundary
	return perimeter_ratio, area_ratio


def dist_pts_ellipse(((ex, ey), (dx, dy), angle), points):
    """
    return unsigned euclidian distances of points to ellipse
    """
    pts = np.float64(points)
    rx, ry = dx / 2., dy / 2.
    angle = (angle / 180.) * np.pi
    # ex,ey =ex+0.000000001,ey-0.000000001 #hack to make 0 divisions possible this is UGLY!!!
    pts = pts - np.array((ex, ey))  # move pts to ellipse appears at origin , with this we copy data -deliberatly!

    M_rot = np.mat([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = np.array(pts * M_rot)  # rotate so that ellipse axis align with coordinate system
    # print "rotated",pts

    pts /= np.array((rx, ry))  # normalize such that ellipse radii=1
    # print "normalize",norm_pts
    norm_mag = np.sqrt((pts * pts).sum(axis=1))
    norm_dist = abs(norm_mag - 1)  # distance of pt to ellipse in scaled space
    # print 'norm_mag',norm_mag
    # print 'norm_dist',norm_dist
    ratio = (norm_dist) / norm_mag  # scale factor to make the pts represent their dist to ellipse
    # print 'ratio',ratio
    scaled_error = np.transpose(pts.T * ratio)  # per vector scalar multiplication: makeing sure that boradcasting is done right
    # print "scaled error points", scaled_error
    real_error = scaled_error * np.array((rx, ry))
    # print "real point",real_error
    error_mag = np.sqrt((real_error * real_error).sum(axis=1))
    # print 'real_error',error_mag
    # print 'result:',error_mag
    return error_mag

def ellipse_true_support(e, raw_edges):
	a, b = e[1][0] / 2., e[1][1] / 2.  # major minor radii of candidate ellipse
	ellipse_circumference = np.pi * abs(3 * (a + b) - np.sqrt(10 * a * b + 3 * (a ** 2 + b ** 2)))
	distances = dist_pts_ellipse(e, raw_edges)
	support_pixels = raw_edges[distances <= 1.3]
	# support_ratio = support_pixel.shape[0]/ellipse_circumference
	return support_pixels, ellipse_circumference

def split_at_corner_index(contour, index):
    """
    contour is array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )
    #index n-2 because the curvature is n-2 (1st and last are not exsistent), this shifts the index (0 splits at first knot!)
    """
    segments = []
    index = [i + 1 for i in index]
    for s, e in zip([0] + index, index + [10000000]):  # list of slice indecies 0,i0,i1,i2,
		segments.append(contour[s:e + 1])  # +1 is for not loosing line segments
    return segments

def ellipse_filter(e):
	in_center = PADDING < e[0][1] < IMG.shape[0] - PADDING and PADDING < e[0][0] < IMG.shape[1] - PADDING
	if in_center:
		is_round = min(e[1]) / max(e[1]) >= MIN_RATIO
		if is_round:
			right_size = PUPIL_MIN <= max(e[1]) <= PUPIL_MAX
			if right_size:
				return True
	return False

def find_kink_and_dir_change(curvature, angle):
    split = []
    if curvature.shape[0] == 0:
        return split
    curv_pos = curvature > 0
    currently_pos = curv_pos[0]
    for idx, c, is_pos in zip(range(curvature.shape[0]), curvature, curv_pos):
        if (is_pos != currently_pos) or abs(c) < angle:
            currently_pos = is_pos
            split.append(idx)
    return split

def GetAnglesPolyline(polyline, closed=False):
    """
    see: http://stackoverflow.com/questions/3486172/angle-between-3-points
    ported to numpy
    returns n-2 signed angles
    """

    points = polyline[:, 0]

    if closed:
        a = np.roll(points, 1, axis=0)
        b = points
        c = np.roll(points, -1, axis=0)
    else:
        a = points[0:-2]  # all "a" points
        b = points[1:-1]  # b
        c = points[2:]  # c points
    # ab =  b.x - a.x, b.y - a.y
    ab = b - a
    # cb =  b.x - c.x, b.y - c.y
    cb = b - c
    # float dot = (ab.x * cb.x + ab.y * cb.y); # dot product
    # print 'ab:',ab
    # print 'cb:',cb

    # float dot = (ab.x * cb.x + ab.y * cb.y) dot product
    # dot  = np.dot(ab,cb.T) # this is a full matrix mulitplication we only need the diagonal \
    # dot = dot.diagonal() #  because all we look for are the dotproducts of corresponding vectors (ab[n] and cb[n])
    dot = np.sum(ab * cb, axis=1)  # or just do the dot product of the correspoing vectors in the first place!

    # float cross = (ab.x * cb.y - ab.y * cb.x) cross product
    cros = np.cross(ab, cb)

    # float alpha = atan2(cross, dot);
    alpha = np.arctan2(cros, dot)
    return alpha * (180. / np.pi)  # degrees

def bin_thresholding(image, image_lower=0, image_upper=256):
    binary_img = cv.inRange(image, np.asarray(image_lower), np.asarray(image_upper))
    return binary_img

def concatenate_vectors(sc, s):
	points = [];
	for r in sc[s]:
		points2 = np.concatenate(r)
		points.append(points2);

	points = np.concatenate(points);
	return points;

def ellipse_eval(contours):
	try:
		c = np.concatenate(contours)
		e = cv.fitEllipse(c)
		d = dist_pts_ellipse(e, c)
		fit_variance = np.sum(d ** 2) / float(d.shape[0])
		return fit_variance <= INITIAL_ELLIPSE_FIT_THRESHOLD
	except :
		return False;
	

def pruning_quick_combine(l, fn, seed_idx=None, max_evals=1e20, max_depth=5):
    """
    l is a list of object to quick_combine.
    the evaluation fn should accept idecies to your list and the list
    it should return a binary result on wether this set is good

    this search finds all combinations but assumes:
        that a bad subset can not be bettered by adding more nodes
        that a good set may not always be improved by a 'passing' superset (purging subsets will revoke this)

    if all items and their combinations pass the evaluation fn you get n	2 -1 solutions
    which leads to (2	n - 1) calls of your evaluation fn

    it needs more evaluations than finding strongly connected components in a graph because:
    (1,5) and (1,6) and (5,6) may work but (1,5,6) may not pass evaluation, (n,m) being list idx's

    """
    if seed_idx:
        non_seed_idx = [i for i in range(len(l)) if i not in seed_idx]
    else:
        # start from every item
        seed_idx = range(len(l))
        non_seed_idx = []
    mapping = seed_idx + non_seed_idx
    unknown = [[node] for node in range(len(seed_idx))]
    # print mapping
    results = []
    prune = []
    while unknown and max_evals:
        path = unknown.pop(0)
        max_evals -= 1
        # print '@idx',[mapping[i] for i in path]
        # print '@content',path
        if not len(path) > max_depth:
            # is this combination even viable, or did a subset fail already?
            if not any(m.issubset(set(path)) for m in prune):
                # we have not tested this and a subset of this was sucessfull before
                if fn([l[mapping[i]] for i in path]):
                    # yes this was good, keep as solution
                    results.append([mapping[i] for i in path])
                    # lets explore more by creating paths to each remaining node
                    decedents = [path + [i] for i in range(path[-1] + 1, len(mapping)) ]
                    unknown.extend(decedents)
                else:
                    # print "pruning",path
                    prune.append(set(path))
    return results

def final_fitting(c, edges):
	# use the real edge pixels to fit, not the aproximated contours
	support_mask = np.zeros(edges.shape, edges.dtype)
	cv.polylines(support_mask, c, isClosed=False, color=(255, 255, 255), thickness=2)
	# #draw into the suport mast with thickness 2
	new_edges = cv.min(edges, support_mask)
	new_contours = cv.findNonZero(new_edges)
	new_e = cv.fitEllipse(new_contours)
	return new_e, new_contours


def algorithm(imgName):
	MIN_RATIO = 0.3
	PUPIL_MIN = 40.0
	PUPIL_MAX = 150.0
	INITIAL_ELLIPSE_FIT_THRESHOLD = 1.8
	STORNG_PERIMETER_RATIO_RANGE = .8, 1.1
	STRONG_AREA_RATIO_RANGE = .6, 1.1
	FINAL_PERIMETER_RATIO_RANGE = .6, 1.2
	STRONG_PRIOR = None

	COLORED_IMAGE = getROI(cv.imread(imgName))
	IMG = cv.cvtColor(COLORED_IMAGE, cv.COLOR_BGR2GRAY)	
	COARSE_PUPIL_WIDTH = IMG.shape[0] / 2
	PADDING = COARSE_PUPIL_WIDTH / 4
	hist = cv.calcHist([IMG], [0], None, [256], [0, 256]); 
	bins = np.arange(hist.shape[0])
	spikes = bins[hist[:, 0] > 40]

	lowest_spike = 0; 
	highest_spike = 0;
	if spikes.shape[0] > 0:
		lowest_spike = spikes.min()
		highest_spike = spikes.max()
	else:
		lowest_spike = 200 
		highest_spike = 255

	# print lowest_spike; 
	# print highest_spike;

	offset = 11; 
	spectral_offset = 5;
	bin_thresh = lowest_spike;

	binary_img = bin_thresholding(IMG, image_upper=lowest_spike + offset)
	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
	cv.dilate(binary_img, kernel, binary_img, iterations=2)
	spec_mask = bin_thresholding(IMG, image_upper=highest_spike - spectral_offset)
	cv.erode(spec_mask, kernel, spec_mask, iterations=1)
	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))

	# open operation to remove eye lashes
	IMG = cv.morphologyEx(IMG, cv.MORPH_OPEN, kernel)

	blur_flag = 1
	# if blur_flag = 1:
	# IMG = cv.medianBlur(IMG,self.blur.value)

	edges = cv.Canny(IMG, 159, 159 * 2 , apertureSize=5)

	# remove edges in areas not dark enough and where the glint is (spectral refelction from IR leds)
	edges = cv.min(edges, spec_mask)
	edges = cv.min(edges, binary_img)

	# get raw edge pix for later
	raw_edges = cv.findNonZero(edges)

	# print raw_edges;

	edges_clone = edges.copy();
	# from edges to contours
	contours, hierarchy = cv.findContours(edges_clone, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE , offset=(0, 0))  # TC89_KCOS
	# contours is a list containing array([[[108, 290]],[[111, 290]]], dtype=int32) shape=(number of points,1,dimension(2) )

	# ## first we want to filter out the bad stuff
	# to short
	good_contours = [c for c in contours if c.shape[0] > 80]

	# now we learn things about each contour through looking at the curvature.
	# For this we need to simplyfy the contour so that pt to pt angles become more meaningfull
	aprox_contours = [cv.approxPolyDP(c, epsilon=1.5, closed=False) for c in good_contours]

	        # if self._window:
	        #   x_shift = COARSE_PUPIL_WIDTH*2
	        #     color = zip(range(0,250,15),range(0,255,15)[::-1],range(230,250))
	split_contours = []

	for c in aprox_contours:
		curvature = GetAnglesPolyline(c)
		# we split whenever there is a real kink (abs(curvature)<right angle) or a change in the genreal direction
		kink_idx = find_kink_and_dir_change(curvature, 80)
		segs = split_at_corner_index(c, kink_idx)

	    #     #TODO: split at shart inward turns
		for s in segs:
			if s.shape[0] > 2:
				split_contours.append(s)
	       	# if self._window:
	        # c = color.pop(0)
	        # color.append(c)
	        # s = s.copy()
	         #   s[:,:,0] += debug_img.shape[1]-COARSE_PUPIL_WIDTH*2
	         #   s[:,:,0] += x_shift
	         #   x_shift += 5
	         #   cv2.polylines(debug_img,[s],isClosed=False,color=map(lambda x: x,c),thickness = 1,lineType=4)#cv2.CV_AA

	split_contours.sort(key=lambda x:-x.shape[0])


	# print [x.shape[0]for x in split_contours]
	if len(split_contours) == 0:
		print "No pupil found"
	    # not a single usefull segment found -> no pupil found
	    # self.confidence.value = 0
	    # self.confidence_hist.append(0)
	    # if self._window:
	    # self.gl_display_in_window(debug_img)
	    # return {'timestamp':frame.timestamp,'norm_pupil':None}

	# removing stubs makes combinatorial search feasable
	split_contours = [c for c in split_contours if c.shape[0] > 3]


	# finding poential candidates for ellipse seeds that describe the pupil.
	strong_seed_contours = []
	weak_seed_contours = []
	for idx, c in enumerate(split_contours):
		if c.shape[0] >= 5:
			e = cv.fitEllipse(c)
		# is this ellipse a plausible canditate for a pupil?
		if ellipse_filter(e):
			distances = dist_pts_ellipse(e, c)
			fit_variance = np.sum(distances ** 2) / float(distances.shape[0])
		if fit_variance <= INITIAL_ELLIPSE_FIT_THRESHOLD:
			# how much ellipse is supported by this contour?
			perimeter_ratio, area_ratio = ellipse_support_ratio(e, [c])
			# logger.debug('Ellipse no %s with perimeter_ratio: %s , area_ratio: %s'%(idx,perimeter_ratio,area_ratio))
		if STORNG_PERIMETER_RATIO_RANGE[0] <= perimeter_ratio <= STORNG_PERIMETER_RATIO_RANGE[1] and STRONG_AREA_RATIO_RANGE[0] <= area_ratio <= STRONG_AREA_RATIO_RANGE[1]:
			strong_seed_contours.append(idx)
			# if self._window:
			cv.polylines(COLORED_IMAGE, [c], isClosed=False, color=(255, 100, 100), thickness=4)
			e = (e[0][0] + COLORED_IMAGE.shape[1] - COARSE_PUPIL_WIDTH * 4, e[0][1]), e[1], e[2]
			cv.ellipse(COLORED_IMAGE, e, color=(255, 100, 100), thickness=3)
		else:
			weak_seed_contours.append(idx)
			# if self._window:
			# cv.polylines(COLORED_IMAGE,[c],isClosed=False,color=(255,255,0),thickness=3)
			# e = (e[0][0]+COLORED_IMAGE.shape[1]-COARSE_PUPIL_WIDTH*4,e[0][1]),e[1],e[2]
			cv.ellipse(COLORED_IMAGE, e, color=(255, 0, 0))
	
	sc = np.array(split_contours)
	# print sc;
	if strong_seed_contours:
		seed_idx = strong_seed_contours
	elif weak_seed_contours:
		seed_idx = weak_seed_contours

	if not (strong_seed_contours or weak_seed_contours):
		print "we can't get an ellipse"
		# if self._window:
		# self.gl_display_in_window(debug_img)
		# self.confidence.value = 0
		# self.confidence_hist.append(0)
		# return {'timestamp':frame.timestamp,'norm_pupil':None}

	solutions = pruning_quick_combine(split_contours, ellipse_eval, seed_idx, max_evals=1000, max_depth=5)
	solutions = filter_subsets(solutions)
	ratings = []

	points = concatenate_vectors(sc, solutions); 

	u_r = []
	for s in solutions:
		e = cv.fitEllipse(concatenate_vectors(sc, s))	
		# if self._window:
		cv.ellipse(COLORED_IMAGE, e, color=(0, 150, 100))
		support_pixels, ellipse_circumference = ellipse_true_support(e, raw_edges)
		support_ratio = support_pixels.shape[0] / ellipse_circumference
		# TODO: refine the selection of final canditate
		if support_ratio >= FINAL_PERIMETER_RATIO_RANGE[0] and ellipse_filter(e):
			ratings.append(support_pixels.shape[0])
		if support_ratio >= STORNG_PERIMETER_RATIO_RANGE[0]:
			# STRONG_PRIOR = u_r.add_vector(p_r.add_vector(e[0])),e[1],e[2]
			# if self._window:
			cv.ellipse(COLORED_IMAGE, e, (0, 255, 255), thickness=6)
		else:
			# not a valid solution, bad rating
			ratings.append(-1)

	# selected ellipse
	if max(ratings) == -1:
		# no good final ellipse found
		print "no good final ellipse found"

	best = solutions[ratings.index(max(ratings))]
	e = cv.fitEllipse(concatenate_vectors(sc, best))

	# final calculation of goodness of fit
	support_pixels, ellipse_circumference = ellipse_true_support(e, raw_edges)
	support_ratio = support_pixels.shape[0] / ellipse_circumference
	goodness = min(1., support_ratio)

	# final fitting and return of result
	new_e, final_edges = final_fitting(sc[best], edges)
	print new_e[0];

	
	# r = cv.rectangle(COLORED_IMAGE, new_e[0][0], new_e[0][1], 0, 0,  (255, 255, 255), -1)
	size_dif = abs(1 - max(e[1]) / max(new_e[1]))
	if ellipse_filter(new_e) and size_dif < .3:
		# if self._window:
		cv.ellipse(IMG, new_e, (0, 255, 255),thickness= 4)
		e = new_e

	cv.imshow("final", COLORED_IMAGE)

	# cv.imshow("image", edges); 
	cv.waitKey(0);
	pass


def capture():
	capture = cv.VideoCapture(0)
	while(True):
		# global IMG;
		# global COLORED_IMAGE; 
		# global COARSE_PUPIL_WIDTH; 
		# global PADDING; 
		ret, frame = capture.read()
		cv.imshow("video", frame)
		COLORED_IMAGE = getROI(frame)
		IMG = cv.cvtColor(COLORED_IMAGE, cv.COLOR_BGR2GRAY)

		COARSE_PUPIL_WIDTH = IMG.shape[0] / 2
		PADDING = COARSE_PUPIL_WIDTH / 4
		algorithm()
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	capture.release()
	cv.destroyAllWindows()
	pass



algorithm("2-eye.png")

# counter = 0;
# while counter < 4:
# 	imgName = str(counter) + "-eye.png";
# 	print imgName;
# 	algorithm(imgName);
# 	counter = counter+1;
# 	pass; 






