import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
import json
from typing import Literal, Union
from mmdet.apis import init_detector, inference_detector

from config import cfg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main')))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))

# from common.utils.inference_utils import process_mmdet_results, non_max_suppression
# from common.base import Demoer
# from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
# from common.utils.vis import render_mesh, save_obj
# from common.utils.human_models import smpl_x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1, dest='num_gpus')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--pretrained_model', type=str, default='smpler_x_b32')
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--agora_benchmark', type=str, default='na')
    # parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--img_path', type=str, default='input', help='input image folder')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--demo_dataset', type=str, default='na')
    parser.add_argument('--demo_scene', type=str, default='all')
    parser.add_argument('--show_verts', action="store_true")
    parser.add_argument('--show_bbox', action="store_true")
    parser.add_argument('--save_mesh', action="store_true")
    parser.add_argument('--multi_person', action="store_true")
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--bbox_thr', type=int, default=50)
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    args.img_path = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\signLangWord\WLASL_images'


    config_path = osp.join('./config', f'config_{args.pretrained_model}.py')
    ckpt_path = osp.join('../pretrained_models', f'{args.pretrained_model}.pth.tar')
    cfg.get_config_fromfile(config_path)

    cfg.update_test_config(args.testset, args.agora_benchmark, shapy_eval_split=None, 
                            pretrained_model_path=ckpt_path, use_cache=False)
    cfg.update_config(args.num_gpus, args.exp_name)
    cudnn.benchmark = True
    # print('Config.trainset_3d:', cfg.trainset_3d)
    # print('Config.human_model_path:', cfg.human_model_path)

    
    from common.utils.inference_utils import process_mmdet_results, non_max_suppression
    from common.base import Demoer
    from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
    from common.utils.vis import render_mesh_open3d, save_obj #, render_mesh
    from common.utils.human_models import smpl_x

    # load model

    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()
    
    start = int(args.start)
    end = start + int(args.end)
    multi_person = args.multi_person
            
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    ### mmdet init
    checkpoint_file = '../pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    config_file= '../pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py'
    model = init_detector(config_file, checkpoint_file, device=device)  # or device='cuda:0'
    print('mmdet model loaded')
    # for frame in tqdm(range(start, end)):
    #     img_path = os.path.join(args.img_path, f'{int(frame):06d}.jpg')
    # viewer = VisMeshPoints()
    
    img_names = os.listdir(args.img_path)
    for i, img_name in enumerate(img_names):
        print(f'Processing {i}/{len(img_names)}, {img_name}')
        img_name_prefix = img_name.split('.')[0]
        img_path = os.path.join(args.img_path, img_name)
        # prepare input image
        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        os.makedirs(args.output_folder, exist_ok=True)

        ## mmdet inference
        mmdet_results = inference_detector(model, img_path)
        mmdet_box = process_mmdet_results(mmdet_results, cat_id=0, multi_person=True)
        # print('mmdet_results', mmdet_results)
        # save original image if no bbox
        if len(mmdet_box[0])<1:
            # save rendered image
            frame_name = img_path.split('/')[-1]
            # save_path_img = os.path.join(args.output_folder, 'img')
            save_path_img = os.path.join(cfg.result_dir, 'img')
            os.makedirs(save_path_img, exist_ok= True)
            cv2.imwrite(os.path.join(save_path_img, f'{frame_name}'), vis_img[:, :, ::-1])
            continue
        
        if not multi_person:
            # only select the largest bbox
            num_bbox = 1
            mmdet_box = mmdet_box[0]
        else:
            # keep bbox by NMS with iou_thr
            mmdet_box = non_max_suppression(mmdet_box[0], args.iou_thr)
            num_bbox = len(mmdet_box)
        
        ## loop all detected bboxes
        for bbox_id in range(num_bbox):
            print(f'Processing bbox {bbox_id}/{num_bbox}')
            mmdet_box_xywh = np.zeros((4))
            mmdet_box_xywh[0] = mmdet_box[bbox_id][0]
            mmdet_box_xywh[1] = mmdet_box[bbox_id][1]
            mmdet_box_xywh[2] =  abs(mmdet_box[bbox_id][2]-mmdet_box[bbox_id][0])
            mmdet_box_xywh[3] =  abs(mmdet_box[bbox_id][3]-mmdet_box[bbox_id][1]) 

            # skip small bboxes by bbox_thr in pixel
            if mmdet_box_xywh[2] < args.bbox_thr or mmdet_box_xywh[3] < args.bbox_thr * 3:
                continue

            # for bbox visualization 
            start_point = (int(mmdet_box[bbox_id][0]), int(mmdet_box[bbox_id][1]))
            end_point = (int(mmdet_box[bbox_id][2]), int(mmdet_box[bbox_id][3]))   

            bbox = process_bbox(mmdet_box_xywh, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img.copy(), bbox, 1.0, 0.0, False, cfg.input_img_shape)
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')
            vertice = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            ## save mesh
            if args.save_mesh:
                # save_path_mesh = os.path.join(args.output_folder, 'mesh')
                save_path_mesh = os.path.join(cfg.result_dir, 'mesh')
                # print('save_path_mesh', save_path_mesh)
                os.makedirs(save_path_mesh, exist_ok= True)
                save_obj(vertice, smpl_x.face, os.path.join(save_path_mesh, f'{img_name_prefix}_{bbox_id}.obj'))

            ## save single person param
            smplx_pred = {}
            smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['leye_pose'] = np.zeros((1, 3))
            smplx_pred['reye_pose'] = np.zeros((1, 3))
            smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10).cpu().numpy()
            smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10).cpu().numpy()
            smplx_pred['transl'] =  out['cam_trans'].reshape(-1,3).cpu().numpy()
            # save_path_smplx = os.path.join(args.output_folder, 'smplx')
            save_path_smplx = os.path.join(cfg.result_dir, 'smplx')
            os.makedirs(save_path_smplx, exist_ok= True)

            npz_path = os.path.join(save_path_smplx, f'{img_name_prefix}_{bbox_id}.npz')
            # print('npz_path', npz_path)
            np.savez(npz_path, **smplx_pred)

            ## render single person mesh
            focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
            princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
            # vis_img = render_mesh(vis_img, vertice, smpl_x.face, {'focal': focal, 'princpt': princpt}, 
            #                       mesh_as_vertices=args.show_verts)
            vis_img = render_mesh_open3d(vis_img, vertice, smpl_x.face, {'focal': focal, 'princpt': princpt}, 
                                  mesh_as_vertices=args.show_verts, bbox=bbox)
            # perspective_projection = render_mesh_open3d(vis_img, vertice, smpl_x.face, {'focal': focal, 'princpt': princpt}, 
            #                       mesh_as_vertices=True, bbox=bbox)
            
            # vis_img = viewer.vis_mesh(vertice)
            
            if args.show_bbox:
                vis_img = cv2.rectangle(vis_img, start_point, end_point, (255, 0, 0), 2)

            ## save single person meta
            meta = {'focal': focal, 
                    'princpt': princpt, 
                    'bbox': bbox.tolist(), 
                    'bbox_mmdet': mmdet_box_xywh.tolist(), 
                    'bbox_id': bbox_id,
                    'img_path': img_path}
            json_object = json.dumps(meta, indent=4)

            # save_path_meta = os.path.join(args.output_folder, 'meta')
            save_path_meta = os.path.join(cfg.result_dir, 'meta')
            os.makedirs(save_path_meta, exist_ok= True)
            with open(os.path.join(save_path_meta, f'{img_name_prefix}_{bbox_id}.json'), "w") as outfile:
                outfile.write(json_object)

        ## save rendered image with all person
        # frame_name = img_path.split('/')[-1]
        frame_name = os.path.basename(img_path)
        # save_path_img = os.path.join(args.output_folder, 'img')
        save_path_img = os.path.join(cfg.result_dir, 'vis')
        os.makedirs(save_path_img, exist_ok= True)


        vis_img_filepath = os.path.join(save_path_img, f'{frame_name}')
        cv2.imwrite(vis_img_filepath, vis_img[:, :, ::-1])
        # break


if __name__ == "__main__":
    main()