'''
对于大范围的航拍图像，需要进以下几个优化
1. 需要对视频进行窗口化处理，而不是每次都传播整个视频
2. mask传播到视频后，同步传播到点云中
3. 点云的mask也应当返回来传播到每一帧图像中
4. 如果当前帧新的object 过多，则调用SAM1 进行全图检测。
5. 构建层级关系树，将每个object 映射到不同的层级上。
'''

import argparse
import os
import random

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    mask_chunk_size = 20
    
    mask_chunks = masks_ord.split(mask_chunk_size, dim=0)
    area_chunks = masks_area.split(mask_chunk_size, dim=0)

    iou_matrix = []
    inner_iou_matrix = []

    for i_areas, i_chunk in zip(area_chunks, mask_chunks):
        row_iou_matrix = []
        row_inner_iou_matrix = []
        for j_areas, j_chunk in zip(area_chunks, mask_chunks):
            intersection = torch.logical_and(i_chunk.unsqueeze(1), j_chunk.unsqueeze(0)).sum(dim=(-1, -2))
            union = torch.logical_or(i_chunk.unsqueeze(1), j_chunk.unsqueeze(0)).sum(dim=(-1, -2))
            local_iou_mat = intersection / union 
            row_iou_matrix.append(local_iou_mat)

            row_inter_mat = intersection / i_areas[:, None]
            col_inter_mat = intersection / j_areas[None, :]

            inter = torch.logical_and(row_inter_mat < 0.5, col_inter_mat >= 0.85)

            local_inner_iou_mat = torch.zeros((len(i_areas), len(j_areas)))
            local_inner_iou_mat[inter] = 1 - row_inter_mat[inter] * col_inter_mat[inter]
            row_inner_iou_matrix.append(local_inner_iou_mat)

        row_iou_matrix = torch.cat(row_iou_matrix, dim=1)
        row_inner_iou_matrix = torch.cat(row_inner_iou_matrix, dim=1)
        iou_matrix.append(row_iou_matrix)
        inner_iou_matrix.append(row_inner_iou_matrix)
    iou_matrix = torch.cat(iou_matrix, dim=0)
    inner_iou_matrix = torch.cat(inner_iou_matrix, dim=0)

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep


def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        if isinstance(masks_lvl, tuple):
            masks_lvl = masks_lvl[0]  # If it's a tuple, take the first element
        if len(masks_lvl) == 0:
            masks_new += (masks_lvl,)
            continue
            
        # Check if masks_lvl is a list of dictionaries
        if isinstance(masks_lvl[0], dict):
            seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
            iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
            stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))
        else:
            # If it's a direct list of masks, use them directly
            seg_pred = torch.from_numpy(np.stack(masks_lvl, axis=0))
            # Create default values for cases without iou and stability
            iou_pred = torch.ones(len(masks_lvl))
            stability = torch.ones(len(masks_lvl))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)
        masks_new += (masks_lvl,)
    return masks_new

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_mask(mask,frame_idx,save_dir):
    image_array = (mask * 255).astype(np.uint8)
    # Create image object
    image = Image.fromarray(image_array[0])

    # Save image
    image.save(os.path.join(save_dir,f'{frame_idx:03}.png'))

def save_masks(mask_list,frame_idx,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    if len(mask_list[0].shape) == 3:
        # Calculate dimensions for concatenated image
        total_width = mask_list[0].shape[2] * len(mask_list)
        max_height = mask_list[0].shape[1]
        # Create large image
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img[0] * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))
    else:
        # Calculate dimensions for concatenated image
        total_width = mask_list[0].shape[1] * len(mask_list)
        max_height = mask_list[0].shape[0]
        # Create large image
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))

def save_masks_npy(mask_list,frame_idx,save_dir):
    np.save(os.path.join(save_dir,f"mask_{frame_idx:03}.npy"),np.array(mask_list))
    
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

def make_enlarge_bbox(origin_bbox, max_width,max_height,ratio):
    width = origin_bbox[2]
    height = origin_bbox[3]
    new_box = [max(origin_bbox[0]-width*(ratio-1)/2,0),max(origin_bbox[1]-height*(ratio-1)/2,0)]
    new_box.append(min(width*ratio,max_width-new_box[0]))
    new_box.append(min(height*ratio,max_height-new_box[1]))
    return new_box

def sample_points(masks, enlarge_bbox,positive_num=1,negtive_num=40):
    ex, ey, ewidth, eheight = enlarge_bbox
    positive_count = positive_num
    negtive_count = negtive_num
    output_points = []
    while True:
        x = int(np.random.uniform(ex, ex + ewidth))
        y = int(np.random.uniform(ey, ey + eheight))
        if masks[y][x]==True and positive_count>0:
            output_points.append((x,y,1))
            positive_count-=1
        elif masks[y][x]==False and negtive_count>0:
            output_points.append((x,y,0))
            negtive_count-=1
        if positive_count == 0 and negtive_count == 0:
            break

    return output_points

def sample_points_from_mask(mask):
    # Get indices of all True values
    true_indices = np.argwhere(mask)

    # Check if there are any True values
    if true_indices.size == 0:
        raise ValueError("The mask does not contain any True values.")

    # Randomly select a point from True value indices
    random_index = np.random.choice(len(true_indices))
    sample_point = true_indices[random_index]

    return tuple(sample_point)

def search_new_obj(masks_from_prev, mask_list,other_masks_list=None,mask_ratio_thresh=0,ratio=0.5, area_threash = 5000):
    new_mask_list = []

    # Calculate mask_none, representing areas not included in any previous masks
    mask_none = ~masks_from_prev[0].copy()[0]
    for prev_mask in masks_from_prev[1:]:
        mask_none &= ~prev_mask[0]

    for mask in mask_list:
        seg = mask['segmentation']
        if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
            new_mask_list.append(mask)
    
    for mask in new_mask_list:
        mask_none &= ~mask['segmentation']
    logger.info(len(new_mask_list))
    logger.info("now ratio:",mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) )
    logger.info("expected ratios:",mask_ratio_thresh)
    if other_masks_list is not None:
        for mask in other_masks_list:
            if mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) > mask_ratio_thresh: # Still a lot of gaps, greater than current thresh
                seg = mask['segmentation']
                if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
                    new_mask_list.append(mask)
                    mask_none &= ~seg
            else:
                break
    logger.info(len(new_mask_list))

    return new_mask_list

def get_bbox_from_mask(mask):
    # Get row and column indices of non-zero elements
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Find min and max indices of non-zero rows and columns
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Calculate width and height
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    
    return xmin, ymin, width, height

def cal_no_mask_area_ratio(out_mask_list):
    h = out_mask_list[0].shape[1]
    w = out_mask_list[0].shape[2]
    mask_none = ~out_mask_list[0].copy()
    for prev_mask in out_mask_list[1:]:
        mask_none &= ~prev_mask
    return(mask_none.sum() / (h * w))


class Prompts:
    def __init__(self,bs:int):
        self.batch_size = bs
        self.prompts = {}
        self.obj_list = []
        self.key_frame_list = []
        self.key_frame_obj_begin_list = []

    def add(self,obj_id,frame_id,mask):
        if obj_id not in self.obj_list:
            new_obj = True
            self.prompts[obj_id] = []
            self.obj_list.append(obj_id)
        else:
            new_obj = False
        self.prompts[obj_id].append((frame_id,mask))
        if frame_id not in self.key_frame_list and new_obj:
            # import ipdb; ipdb.set_trace()
            self.key_frame_list.append(frame_id)
            self.key_frame_obj_begin_list.append(obj_id)
            logger.info("key_frame_obj_begin_list:",self.key_frame_obj_begin_list)
    
    def get_obj_num(self):
        return len(self.obj_list)
    
    def __len__(self):
        if self.obj_list % self.batch_size == 0:
            return len(self.obj_list) // self.batch_size
        else:
            return len(self.obj_list) // self.batch_size +1
    
    def __iter__(self):
        # self.batch_index = 0
        self.start_idx = 0
        self.iter_frameindex = 0
        return self

    def __next__(self):
        if self.start_idx < len(self.obj_list):
            if self.iter_frameindex == len(self.key_frame_list)-1:
                end_idx = min(self.start_idx+self.batch_size, len(self.obj_list))
            else:
                if self.start_idx+self.batch_size < self.key_frame_obj_begin_list[self.iter_frameindex+1]:
                    end_idx = self.start_idx+self.batch_size
                else:
                    end_idx =  self.key_frame_obj_begin_list[self.iter_frameindex+1]
                    self.iter_frameindex+=1
                # end_idx = min(self.start_idx+self.batch_size, self.key_frame_obj_begin_list[self.iter_frameindex+1])
            batch_keys = self.obj_list[self.start_idx:end_idx]
            batch_prompts = {key: self.prompts[key] for key in batch_keys}
            self.start_idx = end_idx
            return batch_prompts
        # if self.batch_index * self.batch_size < len(self.obj_list):
        #     start_idx = self.batch_index * self.batch_size
        #     end_idx = min(start_idx + self.batch_size, len(self.obj_list))
        #     batch_keys = self.obj_list[start_idx:end_idx]
        #     batch_prompts = {key: self.prompts[key] for key in batch_keys}
        #     self.batch_index += 1
        #     return batch_prompts
        else:
            raise StopIteration
        
def get_video_segments(prompts_loader,predictor,inference_state,final_output=False):

    video_segments = {}
    for batch_prompts in tqdm(prompts_loader,desc="processing prompts\n"):
        predictor.reset_state(inference_state)
        for id, prompt_list in batch_prompts.items():
            for prompt in prompt_list:
                # import ipdb; ipdb.set_trace()
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt[0],
                    obj_id=id,
                    mask=prompt[1]
                )
        # start_frame_idx = 0 if final_output else None
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = { }
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
        
        if final_output:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
    return video_segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True)
    parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--level",choices=['default','small','middle','large'])
    parser.add_argument("--batch_size",type=int,default=20)
    parser.add_argument("--sam1_checkpoint",type=str,default="/hpc2hdd/home/qyao951/sam2/checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam2_checkpoint",type=str,default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg",type=str,default="configs/sam2.1/sam2.1_hiera_l.yaml")
    
    parser.add_argument("--detect_stride",type=int,default=10)
    parser.add_argument("--use_other_level",type=int,default=1)
    parser.add_argument("--postnms",type=int,default=0)
    parser.add_argument("--pred_iou_thresh",type=float,default=0.7)
    parser.add_argument("--box_nms_thresh",type=float,default=0.7)
    parser.add_argument("--stability_score_thresh",type=float,default=0.85)
    parser.add_argument("--reverse", action="store_true")
    
    level_dict = {
        "default": 0,
        "small": 1, 
        "middle": 2,
        "large": 3
    }
    args = parser.parse_args()
    logger.add(os.path.join(args.output_dir,f'{args.level}.log'), rotation="500 MB")
    logger.info(args)
    video_dir = args.video_path
    level = args.level
    base_dir = args.output_dir

    ##### load Sam2 and Sam1 Model #####
    sam2_checkpoint = args.sam2_checkpoint
    model_cfg = args.sam2_model_cfg
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)

    sam_ckpt_path = args.sam1_checkpoint

    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=args.pred_iou_thresh, 
        box_nms_thresh=args.box_nms_thresh, 
        stability_score_thresh=args.stability_score_thresh, 
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        # min_mask_region_area=100,
    )
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    try:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]), reverse=args.reverse)
    except:
        frame_names.sort(key=lambda p: os.path.splitext(p)[0], reverse=args.reverse)

    now_frame = 0
    inference_state = predictor.init_state(video_path=video_dir)
    masks_from_prev = []
    sum_id = 0 # Record total number of objects

    prompts_loader = Prompts(bs=args.batch_size)  # hold all the clicks we add for visualization
    
    while True:
        logger.info(f"frame: {now_frame}")
        
        sum_id = prompts_loader.get_obj_num()
        image_path = os.path.join(video_dir,frame_names[now_frame])
        image = cv2.imread(image_path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize if the input is too large:
        orig_h, orig_w = image.shape[:2]
        if orig_h > 1080:
            logger.info("Resizing original image to 1080P...")
            scale = 512 / orig_h
            h = int(orig_h * scale)
            w = int(orig_w * scale)
            image = cv2.resize(image, (w, h))

        # Generate only large masks
        # masks_l = mask_generator.generate_l(image)
        # all_masks = mask_generator.generate(image) # 使用sam1 进行分割当前帧
        # with torch.autocast("cuda", dtype=torch.float32): # 原因 PyTorch Version: 1.13.0 版本，不支持直接转换 BFloat16 格式 // torch   2.7.0+cu118
        all_masks = mask_generator.generate(image) # 使用sam1 进行分割当前帧
        masks = all_masks[level_dict[args.level]]
        # masks_l = mask_generator.generate(image)
        
        if args.postnms:
        #     # Pass masks_l directly, no need to wrap in tuple
        #     # masks_l = masks_update(masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)[0]
            masks = masks_update(masks, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)[0]

        # Use large level masks
        # masks = masks_l
        other_masks = None
        
        if not args.use_other_level:
            other_masks = None

        if now_frame == 0: # first frame
            ann_obj_id_list = range(len(masks))
            for ann_obj_id in tqdm(ann_obj_id_list):
                seg = masks[ann_obj_id]['segmentation']
                prompts_loader.add(ann_obj_id,0,seg)
        else:  
            new_mask_list = search_new_obj(masks_from_prev, masks, other_masks,mask_ratio_thresh) # 如果不是第一帧，则和前面的帧进行对比，找到新的obj
            logger.info(f"number of new obj: {len(new_mask_list)}")

            for id,mask in enumerate(masks_from_prev):
                if mask.sum() == 0:
                    continue
                prompts_loader.add(id,now_frame,mask[0]) # 把前面帧的 存在在当前帧的mask 都放在prompts中

            for i in range(len(new_mask_list)):
                new_mask = new_mask_list[i]['segmentation']
                prompts_loader.add(sum_id+i,now_frame,new_mask) # 把新的mask 加到prompts中

        logger.info(f"obj num: {prompts_loader.get_obj_num()}")

        if now_frame==0 or len(new_mask_list)!=0:
            video_segments = get_video_segments(prompts_loader,predictor,inference_state) #使用前面的和当前mask 分割整个视频
        
        vis_frame_stride = args.detect_stride
        max_area_no_mask = (0,-1)
        for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
            if out_frame_idx < now_frame:
                continue
            
            out_mask_list = []
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                out_mask_list.append(out_mask)
            
            no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
            if now_frame == out_frame_idx:
                mask_ratio_thresh = no_mask_ratio
                logger.info(f"mask_ratio_thresh: {mask_ratio_thresh}")

            if no_mask_ratio > mask_ratio_thresh + 0.01 and out_frame_idx > now_frame:
                masks_from_prev = out_mask_list
                max_area_no_mask = (no_mask_ratio, out_frame_idx)
                logger.info(max_area_no_mask)
                break
        if max_area_no_mask[1] == -1:
            break
        logger.info("max_area_no_mask:", max_area_no_mask)
        now_frame = max_area_no_mask[1]

    ###### Final output ######
    save_dir = os.path.join(base_dir,level,"final-output")
    os.makedirs(save_dir, exist_ok=True)
    
    video_segments = get_video_segments(prompts_loader,predictor,inference_state,final_output=True)
    for out_frame_idx in tqdm(range(0, len(frame_names), 1)):
        out_mask_list = []
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask_list.append(out_mask)

        no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
        logger.info(no_mask_ratio)

        save_masks(out_mask_list, out_frame_idx,save_dir)
        save_masks_npy(out_mask_list, out_frame_idx,save_dir)

    ###### Generate Visualization Frames ######
    logger.info("Start generating visualization frames...")
    vis_save_dir = os.path.join(base_dir,level,'visualization','full-mask-npy')
    os.makedirs(vis_save_dir,exist_ok=True)
    frame_save_dir = os.path.join(base_dir,level,'visualization','frames')
    os.makedirs(frame_save_dir, exist_ok=True)

    # Read all npy files
    npy_name_list = []
    for name in os.listdir(save_dir):
        if 'npy' in name:
            npy_name_list.append(name)
    npy_name_list.sort()
    logger.info(f"Found {len(npy_name_list)} npy files")

    npy_list = [np.load(os.path.join(save_dir,name)) for name in npy_name_list]
    image_list = [Image.open(os.path.join(video_dir,name)) for name in frame_names]

    assert len(npy_list) == len(image_list), "Number of npy files does not match number of images"
    logger.info(f"Processing {len(npy_list)} frames in total")

    # Generate random colors
    def generate_random_colors(num_colors):
        colors = []
        for _ in range(num_colors):
            reroll = True
            iter_cnt = 0
            while reroll and iter_cnt < 100:
                iter_cnt += 1
                reroll = False
                color = tuple(random.randint(1, 255) for _ in range(3))
                for selected_color in colors:
                    if np.linalg.norm(np.array(color) - np.array(selected_color)) < 70:
                        reroll = True
                        break
            colors.append(color)
        return colors

    num_masks = max(len(masks) for masks in npy_list)
    colors = generate_random_colors(num_masks)
    post_colors = [(0, 0, 0)] + colors
    post_colors = np.array(post_colors) # [num_masks, 3]
    np.save(os.path.join(base_dir, "colors.npy"), post_colors)

    # Only process first and last frames
    # frames_to_process = [0, -1]  # Indices for first and last frames
    
    for frame_idx in range(len(frame_names)):
    # for frame_idx in frames_to_process:
        masks = npy_list[frame_idx]
        image = image_list[frame_idx]
        
        image_np = np.array(image)
        mask_combined = np.zeros_like(image_np, dtype=np.uint8)

        for mask_id, mask in enumerate(masks):
            mask = mask.squeeze(0)
            mask_area = mask > 0
            mask_combined[mask_area, :] = colors[mask_id]

        # Blend original image with colored mask
        mask_combined = np.clip(mask_combined, 0, 255)

        # blended_image = cv2.addWeighted(image_np, 0.7, mask_combined, 0.3, 0)
        blended_image = mask_combined

        # change the save path
        frame_name = frame_names[frame_idx]
        frame_save_dir = base_dir
        output_path = os.path.join(frame_save_dir, frame_name)
        Image.fromarray(blended_image).save(output_path)
        logger.info(f"Frame saved to: {output_path}")