import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip
from dataset.dataset_TM_eval_test import OOD_Dataset
import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
# import pydevd_pycharm
# pydevd_pycharm.settrace('10.8.32.196', port=19999, stdoutToServer=True, stderrToServer=True)
##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)
from tqdm import tqdm
args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Network ---- #####

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code,
                                embed_dim=args.embed_dim_gpt,
                                clip_dim=args.clip_dim,
                                block_size=args.block_size,
                                num_layers=args.num_layers,
                                n_head=args.n_head_gpt,
                                drop_out_rate=args.drop_out_rate,
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

data = OOD_Dataset(motion_dir=None, motion_length=64)

with torch.no_grad():
    for keyid in range(len(data)):
        _, caption, _ = data.load_keyid(keyid)
        # get clip features
        text = clip.tokenize(caption, truncate=True).cuda()
        feat_clip_text = clip_model.encode_text(text).float()
        clip_feat = clip_feat.permute(1, 0, 2)

        try:
            index_motion = trans_encoder.sample(feat_clip_text, True)
        except:
            index_motion = torch.ones(1, 1).cuda().long()

        pred_pose = net.forward_decoder(index_motion)
        os.makedirs(os.path.join(args.out_dir, str(keyid).zfill(5)), exist_ok=True)
        np.save(os.path.join(args.out_dir, str(keyid).zfill(5), 'motion.npy'), pred_pose.detach().cpu().numpy())




