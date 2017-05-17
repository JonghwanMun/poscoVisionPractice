import os
import h5py
import json
import random
import numpy as np
from PIL import Image

def load_json(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    return json_file

def write_json(json_path, json_file):
    with open(json_path, 'w') as f:
		json.dump(json_file, f)        

class dataLoader():
    def __init__(self, params):

		self.img_dir = params['img_dir']

		# load numpy data
		h5py_file = h5py.File(params['h5py_path'], 'r')
		self.caption_labels = h5py_file['/caption_labels']
		self.caption_length = h5py_file['/caption_length']
		#h5py_file.close()

		# load feature hdf5
		if 'prepared_feature' in params.keys():
			self.prepared_feature = params['prepared_feature']
			self.feat_file = h5py.File(params['feature_path'], 'r')
		else:
			self.prepared_feature = False

		# load json data
		json_file = load_json(params['json_path'])
		self.wtoi = json_file['wtoi']
		self.itow = json_file['itow']
		self.caption_ids = json_file['caption_ids']
		self.image_ids = json_file['image_ids']
		self.image_path = json_file['image_path']
		self.max_caption_length = json_file['max_caption_length']
		self.dim_image= 4096

		# set iterator
		self.iter_idx = 0
		self.num_captions = len(self.caption_ids)
		self.vgg_mean = np.array([123.68, 116.779, 103.939])
		self.iter_order = np.arange(self.num_captions)
		np.random.shuffle(self.iter_order)

    def getWtoi(self):
        return self.wtoi

    def getItow(self):
        return self.itow

    def getVocabSize(self):
        return len(self.itow)

    def getNumCaptions(self):
        return self.num_captions

    def getMaxCaptionLength(self):
        return self.max_caption_length

    def resetIterIndex(self):
        self.iter_idx = 0

    def incrementIterIndex(self):
        self.iter_idx += 1

    def getBatch(self, batch_size=50):

		batch_caption_length = np.zeros(batch_size, dtype='uint32')
		batch_caption_labels = np.zeros((batch_size, self.max_caption_length), dtype='uint32')
		batch_target_labels = np.full((batch_size, self.max_caption_length), self.getWtoi()['<E>'], dtype='uint32')
		batch_caption_masks = np.zeros((batch_size, self.max_caption_length), dtype='uint32')
		batch_image_path = []
		batch_image_ids = []

		if self.prepared_feature:
			batch_images_feat = np.zeros((batch_size, self.dim_image), dtype='float')
		else:
			batch_images = np.zeros((batch_size, 224, 224, 3), dtype='float')

		for bi in range(batch_size):
			if self.iter_idx == self.num_captions:
				self.resetIterIndex()

			# load caption label and length
			cap_length = self.caption_length[self.iter_order[self.iter_idx]]
			batch_caption_length[bi] = cap_length
			batch_caption_labels[bi] = self.caption_labels[self.iter_order[self.iter_idx]]
			batch_target_labels[bi, 0:-1] = self.caption_labels[self.iter_order[self.iter_idx], 1:]
			batch_caption_masks[bi, :cap_length] = 1

			# load and preprocessing an image
			if self.prepared_feature:
				batch_images_feat[bi] = self.feat_file[str(self.image_ids[self.iter_order[self.iter_idx]])]
			else:
				img = Image.open(self.img_dir + self.image_path[self.iter_order[self.iter_idx]]).convert('RGB') #(224,224,3)
				img = np.array(img, 'f')
				for i in range(3): img[:,:,i] = img[:,:,i] - self.vgg_mean[i]
				#img = np.transpose(img, [2,0,1]) # (224,224,3) -> (3,224,224)
				batch_images[bi] = img

		# load image id
		batch_image_ids.append(self.image_ids[self.iter_order[self.iter_idx]])
		# increment iterator
		self.incrementIterIndex()

		batch = {}
		batch['caption_labels'] = batch_caption_labels
		batch['target_labels'] = batch_target_labels
		batch['caption_length'] = batch_caption_length
		batch['caption_masks'] = batch_caption_masks
		batch['image_ids'] = batch_image_ids
		if self.prepared_feature:
			batch['images_feat'] = batch_images_feat
		else:
			batch['images'] = batch_images

		return batch
