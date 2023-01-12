import glob
import os
import sys
import utils
import cv2 as cv
import itertools
from tqdm import tqdm
import numpy as np

def main():
	os.chdir('/Users/felixmeng/Desktop/CME291/Code/dataset')

	# images=glob.glob("*.png")
	# print(images)
	test_img='stanford.png'
	img = cv.imread(test_img)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	# k=[0.02,0.03,0.05]
	# p=[0.02,0.04]
	ks,ps=get_k_p()
	with tqdm(total=len(ks)) as pbar:
		for k in ks:
			rad_img=utils.radial_distortion_fixed(img,k)
			pbar.update(1)
	with tqdm(total=len(ps)) as pbar:
		for p in ps:
			tang_img=utils.tangential_distortion_fixed(img,p)
			pbar.update(1)
	rad_img=utils.radial_distortion_fixed(img,k)
def get_k_p():
	k1=np.arange(0,0.1,0.01)
	k2=np.arange(0,0.1,0.01)
	k3=np.arange(0,0.1,0.01)
	p1=np.arange(0,0.1,0.01)
	p2=np.arange(0,0.1,0.01)
	distorted_k=[k1,k2,k3]
	disorted_ks = list(itertools.product(*distorted_k))
	len(disorted_ks)
	distorted_p=[p1,p2]
	distorted_ps=list(itertools.product(*distorted_p))
	print(disorted_ks,distorted_ps)
	return disorted_ks,distorted_ps
def read_images():
	#os.chdir('/Users/felixmeng/Desktop/CME291/Code')
	#images=glob.glob("*.png")
	for img in images:
		img = cv.imread(test_img)
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		tang_img=utils.tangential_distortion_fixed(img,p)
		rad_img=utils.radial_distortion_fixed(img,p)
if __name__ == "__main__":
	main()