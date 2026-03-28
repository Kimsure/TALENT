import os
import cv2
import lmdb
import math
# import faiss
import numpy as np
import pyarrow as pa
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Union
from refer import refer

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
# import spacy

import albumentations as A
from albumentations.pytorch import ToTensorV2

def normalize(img_w, img_h, box, tail=True):
    xmin, ymin, xmax, ymax = box[:4]
    new_box = [xmin/img_w, ymin/img_h, xmax/img_w, ymax/img_h]
    new_box = np.clip(new_box, 0, 1).tolist()
    if tail and len(box)>=4:
        new_box.extend(box[4:])
    return new_box


info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'easy' : 69,
        'hard' :70,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class RefDataset(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length, args):
        super(RefDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.img_sz = input_size
        self.input_size = (input_size, input_size)
        #self.mask_size = [13, 26, 52]
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.length = info[dataset][split]
        self.env = None
        self.args = args
        self.aug = args.aug
        
        # self.coco_transforms = make_coco_transforms(mode, cautious=False)
        
        each_img_sz = int(input_size/math.sqrt(args.aug.num_bgs))
        mean = (0.48145466, 0.4578275, 0.40821073) #(0.485, 0.456, 0.406)
        std = (0.26862954, 0.26130258, 0.27577711) #(0.229, 0.224, 0.225)
        self.resize_bg1 = A.Compose([
            A.Resize(input_size, input_size, always_apply=True)])
        self.resize_bg4 = A.Compose([
            A.Resize(each_img_sz, each_img_sz, always_apply=True)],
            additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image',
                                'mask1': 'mask', 'mask2': 'mask', 'mask3': 'mask',})
        self.each_img_sz = each_img_sz
        self.transforms = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2 (),
        ])
        

        if self.mode != 'test':
            ## Bringing out logits
            if self.dataset == 'refcoco' :
                self.refer = refer.REFER(dataset='refcoco', splitBy='unc')
            elif self.dataset == 'refcoco+' :
                self.refer = refer.REFER(dataset='refcoco+', splitBy='unc')
            elif self.dataset  =='refcocog_u' :
                self.refer = refer.REFER(dataset='refcocog', splitBy='umd')
            print(f"Bringing out logits of {self.dataset} dataset")


            ## Tools by refer.REFER
            ref_ids = self.refer.getRefIds(split=split)
            self.img_ids = self.refer.getImgIds(ref_ids)
            self.ref_id2idx = dict(zip(ref_ids, range(len(ref_ids)))) # ref_id -> idx(key)
            self.idx2ref_id = dict(zip(range(len(ref_ids)), ref_ids)) # idx(key) -> ref_id
            
            img_ids = self.refer.getImgIds(ref_ids)
            # img_ids.sort()
            self.img_id2idx = dict(zip(img_ids, range(len(img_ids)))) # ref_id -> idx(key)
            self.idx2img_id = dict(zip(range(len(img_ids)), img_ids)) # idx(key) -> ref_id
        else :
            print("Test mode does not require logits of dataset")

        if self.args.aug.blur :
            self.blur = ImageFilter.GaussianBlur(100)
        # index = faiss.IndexFlatIP(512)
        # self.db = faiss.IndexIDMap2(index)

        np.random.seed()

    def _init_db(self):
        
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        if self.args.dataset == 'refcocog_u' :
            img2img_path = '/data2/projects/chaeyun/CRIS_R/logit_db/refcocog_u/refcocog_u_logit_i2i_score.lmdb'
            text2img_path =  '/data2/projects/chaeyun/CRIS_R/logit_db/refcocog_u/refcocog_u_logit_t2i_score_5k.lmdb'
        elif self.args.dataset == 'refcoco' :
            img2img_path = '/data2/projects/chaeyun/CRIS_R/logit_db/refcoco/refcoco_logit_i2i_score_5k.lmdb'
            text2img_path =  '/data2/projects/chaeyun/CRIS_R/logit_db/refcoco/refcoco_logit_t2i_score.lmdb'
        elif self.args.dataset == 'refcoco+' :
            img2img_path = '/data2/projects/chaeyun/CRIS_R/logit_db/refcoco+/refcocop_logit_i2i_score_5k.lmdb'
            text2img_path =  '/data2/projects/chaeyun/CRIS_R/logit_db/refcoco+/refcocop_logit_t2i_score.lmdb'
        

        # default setting : t2i logit
        self.t2i_env = lmdb.open(
            text2img_path, subdir=False, max_readers=32,
            readonly=True, lock=False,
            readahead=False, meminit=False
        )
        with self.t2i_env.begin(write=False) as txn:
            logit_length = loads_pyarrow(txn.get(b'__len__'))
            self.logit_keys = loads_pyarrow(txn.get(b'__keys__'))
        
        # self.nlp = spacy.load("en_core_web_sm")
        
        if not self.aug.t2i_only :            
            self.i2i_env = lmdb.open(
                img2img_path, subdir=False, max_readers=32,
                readonly=True, lock=False,
                readahead=False, meminit=False
            )
            with self.i2i_env.begin(write=False) as txn:
                logit_length_i = loads_pyarrow(txn.get(b'__len__'))
                self.logit_keys_i = loads_pyarrow(txn.get(b'__keys__'))




    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        
        """
        Convert keys to ref_id
        """
        ref_id = self.idx2ref_id[index]
        ref_img_id = self.refer.Refs[ref_id]['image_id']

        t2i_index = self.ref_id2idx[ref_id]
        with self.t2i_env.begin(write=False) as txn:
            t2i_byteflow = txn.get(self.logit_keys[t2i_index])
        t2i_similarity = loads_pyarrow(t2i_byteflow)


        target_sent_idx = np.random.choice(list(range(ref['num_sents'])), size=1, replace=False)[0]
        sent = ref['sents'][target_sent_idx]
                
        if not self.aug.t2i_only :   
            print("Both T2I and I2I")             
            i2i_index = self.img_id2idx[ref_img_id]
            with self.i2i_env.begin(write=False) as txn:
                i2i_byteflow = txn.get(self.logit_keys_i[i2i_index])
            i2i_similarity = loads_pyarrow(i2i_byteflow)
            invalid_choices = set([img_id for img_id, score in i2i_similarity if score > self.args.aug.i2i_sim_thres])
                
        if self.aug.fp_restrict :
            doc = self.nlp(sent)
            target_noun = self.get_target_noun(sent, doc)
            
            left_right = {"left" : [0, 2],
                        "right" : [1, 3]}
            other_directions = {"top" :[0, 1],
                            "high" :[0, 1],
                            "above" :[0, 1],
                            "bottom" :[2, 3],
                            "low" :[2, 3],
                            "below" :[2, 3],
                            "under" :[2, 3],
                            "beneath" :[2, 3]
                            }

            left_right_check = False
            other_pos_check = False 

            for lr in left_right.keys() :
                if lr in sent :
                    left_right_check = True
                    lr_cand = left_right[lr]
                    break
            
            for other in other_directions.keys() :
                if other in sent :
                    other_pos_check = True
                    other_cand = other_directions[other]
                    break       

            pos_check = left_right_check or other_pos_check
            
            
            
        
        """
        Decide Mosaic Size
        """
        ## Train config
        if self.mode == 'train':
            if self.args.aug.num_bgs > 1:
                aug_prob = self.aug.aug_prob
                
                # Before retrieval_epoch: One Image or Random Mosaic
                if self.args.aug.epoch < self.args.aug.retrieval_epoch:
                    if self.args.aug.mix_grid : 
                        num_bgs = np.random.choice([1, 4, 9], p=[1-aug_prob, aug_prob/2, aug_prob/2]) #
                        mosaic_type = 'random'
                    else :
                        num_bgs = np.random.choice([1, 4], p=[1-aug_prob, aug_prob]) #
                        mosaic_type = 'random'
                # Retrieval 
                else:
                    rand_prob = self.aug.rand_prob
                    retr_prob = self.aug.retr_prob                  
                    # After retrieval_epoch: Decide between One Image, Random Mosaic, or Retrieval Based Mosaic
                    choice = np.random.choice(['one', 'random', 'retrieval'], p=[1-(rand_prob + retr_prob), rand_prob, retr_prob])
                    mosaic_type = choice
                    if choice == 'one':
                        num_bgs = 1
                    elif self.args.aug.mix_grid :
                        num_bgs = np.random.choice([4, 9], p=[0.5, 0.5])
                    else :
                        num_bgs = 4
            else:
                num_bgs = 1
                mosaic_type = 'one'
        ## Test, Val Config
        else :
            num_bgs = 1
            mosaic_type ='one'

        ## Choosing dataloader strategy        
        if num_bgs > 1 :
            if mosaic_type == 'retrieval':
                sent_id = list(t2i_similarity.keys())[target_sent_idx]
                if self.aug.t2i_only : 
                    invalid_choices = set([img_id for img_id, score in t2i_similarity[sent_id] if score > self.args.aug.t2i_sim_thres])
                valid_img_score_list = [pair[0] for pair in t2i_similarity[sent_id] if pair[0] not in invalid_choices]
                len_valid = len(valid_img_score_list)

                if len_valid == 0:
                    print(f"Valid image list is 0", len(invalid_choices))
                    valid_img_score_list = [pair[0] for pair in t2i_similarity[sent_id][-300:]]


                if self.args.aug.top_k < 20 :
                    img_ids = list(np.random.choice(valid_img_score_list[:self.args.aug.top_k], size=num_bgs-1, replace=False))
                elif len_valid < self.args.aug.top_k :
                    img_ids = list(np.random.choice(valid_img_score_list, size=num_bgs-1, replace=True))
                else :
                    img_ids = list(np.random.choice(valid_img_score_list[:self.args.aug.top_k], size=num_bgs-1, replace=True))
                    
                ref_int = [self.refer.imgToRefs[i][0]['ref_id'] for i in img_ids]
                keys = [str(self.ref_id2idx[k]).encode('utf-8') for k in ref_int]
                                
            else:  # mosaic_type == 'random'
                # Random Mosaic
                keys = list(np.random.choice(self.keys, size=num_bgs-1, replace=False))
                
            refs = []
            for key in keys:
                with env.begin(write=False) as txn:
                    byteflow = txn.get(key)
                    ref_other = loads_pyarrow(byteflow)
                refs.append(ref_other)
        ## One Image (num_bgs = 1)
        else:
            keys = []
            refs = []

        if num_bgs == 1 :
            insert_idx = np.random.choice(range(num_bgs))
        else : 
            if self.aug.fp_restrict :    
                if left_right_check and other_pos_check :
                    intersection = list(set(lr_cand) & set(other_cand))
                    if intersection : insert_idx = intersection[0]
                elif left_right_check :
                    insert_idx = np.random.choice(lr_cand)
                elif other_pos_check : 
                    insert_idx = np.random.choice(other_cand)
                else : 
                    insert_idx = np.random.choice(range(num_bgs))
            else : 
                insert_idx = np.random.choice(range(num_bgs))

        refs.insert(insert_idx, ref)


        if self.args.aug.tgt_selection == 'fixed':
            target_idx = insert_idx
            
        target_ref = refs[target_idx]
        
        # load items
        imgs, masks, sents_arr, seg_ids = [], [], [], []
        org_img_sizes = []
        for ref in refs:
            
            ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                                cv2.IMREAD_COLOR)
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            org_img_sizes.append(img.shape[:2])
            
            mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                    cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.
            masks.append(mask)
            
            seg_id = ref['seg_id']
            seg_ids.append(seg_id)
            sents_arr.append(ref['sents'])
                
            
        # image resize and apply 4in1 augmentation
        if num_bgs==1:
            resized = self.resize_bg1(image=imgs[0], mask=masks[0])
            imgs, masks = [resized['image']], [resized['mask']]
            img = imgs[0]
        else:
            if self.args.aug.move_crs_pnt:
                if self.args.aug.rand_crs :
                    print("Cross point random shift")
                    crs_y = np.random.randint(0, self.img_sz+1)
                    crs_x = np.random.randint(0, self.img_sz+1)
                else : 
                    print("Cross point shift within range")
                    quarter_sz = self.each_img_sz/2
                    crs_y = np.random.randint(quarter_sz, self.img_sz+1-quarter_sz)
                    crs_x = np.random.randint(quarter_sz, self.img_sz+1-quarter_sz)
            else:
                denum = int(math.sqrt(num_bgs))
                crs_y = self.img_sz//denum
                crs_x = self.img_sz//denum

            imgs_resized = []
            masks_resized = []

            for i in range(num_bgs):
                row = i // denum
                col = i % denum
                y_start = row * crs_y
                y_end = y_start + crs_y
                x_start = col * crs_x
                x_end = x_start + crs_x
                
                if y_end > self.img_sz or x_end > self.img_sz:
                    img_resized = np.zeros([crs_y, crs_x, 3])
                    mask_resized = np.zeros([crs_y, crs_x])
                else:
                    resize_transform = A.Compose([
                        A.Resize(crs_y, crs_x, always_apply=True)
                    ])
                    temp = resize_transform(image=imgs[i], mask=masks[i])
                    img_resized = temp['image']
                    mask_resized = temp['mask']

                imgs_resized.append(img_resized)
                masks_resized.append(mask_resized)

            imgs = imgs_resized
            masks = masks_resized

            
            # scale effect ablation
            if self.args.aug.blur:
                imgs = [np.asarray(Image.fromarray(x).filter(self.blur)) if i!=insert_idx else x for i, x in enumerate(imgs)]
            
            num_rows = num_cols = int(math.sqrt(num_bgs))
            idxs = [(i*num_cols,i*num_cols+num_cols) for i in range(num_rows)]
            img = [np.concatenate(imgs[_from:_to], axis=1) for (_from, _to) in idxs]
            img = np.concatenate(img, axis=0).astype(np.uint8)
            
            masks_arr = []
            for bg_idx in range(num_bgs):
                mask = masks[bg_idx]
                temp = [mask if idx==bg_idx else np.zeros_like(masks[idx]) for idx in range(num_bgs)]
                mask = [np.concatenate(temp[_from:_to], axis=1) for (_from, _to) in idxs]
                mask = np.concatenate(mask, axis=0).astype(np.int32)
                masks_arr.append(mask)
            masks = masks_arr

        mask = masks[target_idx]    
        mask = mask.astype(np.uint8)
        mask[mask>0] = 1
        
        img_size = img.shape[:2]
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])

        sents = sents_arr[target_idx]
        
        if self.mode=='train':
            mask = cv2.warpAffine(mask,
                        mat,
                        self.input_size,
                        flags=cv2.INTER_LINEAR,
                        borderValue=0.)
            sent = sents[target_sent_idx]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img, mask = self.convert(img, mask)
            return img, word_vec, mask
        
        seg_id = seg_ids[target_idx]
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        
        if self.mode == 'val':
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            assert len(org_img_sizes)==1
            params = {
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(org_img_sizes[0])
            }
            return img, word_vec, mask, params
        
        else:
            # sentence -> vector
            img = self.convert(img)[0]
            assert len(org_img_sizes)==1
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(org_img_sizes[0]),
                'sents': sents
            }
            return img, mask, params
        
        return ori_img, img, word_vecs, mask, pad_masks, seg_id, sents,

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask
    

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.lmdb_dir}, " + \
            f"dataset={self.dataset}, " + \
            f"split={self.split}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"



    def get_target_noun(self, sent, doc) :
        get_nouns = [token.text for i, token in enumerate(doc) if token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS'] and token.dep_ not in ['amod','advmod', 'nummod', 'quantmod'] and token.text not in [".", ',', ' ']]
        # get target noun
        target_noun = ""

        for i, token in enumerate(doc) :
            if token.dep_ == 'ROOT' and token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS'] :
                target_noun = token.text
                break
            elif token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS'] and token.dep_ in ['nsubj', 'nsubjpass', 'attr' ,'dep'] :
                target_noun = token.text
                break

        if not target_noun:
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ == 'compound':
                    target_noun = token.text
                    break
                elif token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                    if len(get_nouns) == 1:
                        target_noun = get_nouns[0] 
                    else:
                        target_noun = token.text
                    break
        if not target_noun :
            for token in doc:
                if token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS'] and token.dep_ in ['pobj'] :
                    target_matnoun = token.text
                    break 

        if not target_noun and len(get_nouns) == 1:
            target_noun = get_nouns[0]

        return target_noun