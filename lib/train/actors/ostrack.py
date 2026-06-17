from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        scale_consistency_loss, scale_status = self.compute_mstf_scale_consistency_loss(pred_dict)
        if scale_consistency_loss is not None:
            scale_cfg = self.cfg.TRAIN.MSTF_SCALE_CONSISTENCY
            loss = loss + float(scale_cfg.WEIGHT) * scale_consistency_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            status.update(scale_status)
            return loss, status
        else:
            return loss
    def compute_mstf_scale_consistency_loss(self, pred_dict):
        scale_cfg = getattr(self.cfg.TRAIN, "MSTF_SCALE_CONSISTENCY", None)
        if scale_cfg is None or not getattr(scale_cfg, "ENABLE", False):
            return None, {}

        branch_weights = pred_dict.get("mstf_branch_weights", None)
        scale_probs = pred_dict.get("mstf_scale_probs", None)
        if branch_weights is None and scale_probs is None:
            return None, {}

        pred_boxes = pred_dict["pred_boxes"].detach()
        pred_wh = pred_boxes[..., 2:4].mean(dim=1).clamp_min(1e-6)
        scale_value = torch.sqrt(pred_wh[:, 0] * pred_wh[:, 1])
        target3 = self._soft_scale_target(scale_value, scale_cfg)

        total = pred_boxes.sum() * 0.0
        status = {}
        if branch_weights is not None:
            branch_probs = branch_weights.clamp_min(1e-6)
            branch_probs = branch_probs / branch_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
            branch_loss = F.kl_div(branch_probs.log(), target3, reduction="batchmean")
            total = total + float(getattr(scale_cfg, "BRANCH_WEIGHT", 1.0)) * branch_loss
            status["Loss/mstf_scale_branch"] = branch_loss.item()
            status["MSTF/branch_local"] = branch_probs[:, 0].mean().item()
            status["MSTF/branch_mid"] = branch_probs[:, 1].mean().item()
            status["MSTF/branch_context"] = branch_probs[:, 2].mean().item()

        if scale_probs is not None and float(getattr(scale_cfg, "STATE_WEIGHT", 0.0)) > 0.0:
            uncertain_weight = float(getattr(scale_cfg, "UNCERTAIN_TARGET_WEIGHT", 0.0))
            if uncertain_weight > 0.0:
                scale_confidence = target3.max(dim=1, keepdim=True).values
                uncertain_target = ((1.0 - scale_confidence) * uncertain_weight).clamp(0.0, 0.5)
                state_target = torch.cat([target3 * (1.0 - uncertain_target), uncertain_target], dim=1)
            else:
                state_target = torch.cat([target3, target3.new_zeros(target3.shape[0], 1)], dim=1)
            state_target = state_target / state_target.sum(dim=1, keepdim=True).clamp_min(1e-6)
            state_probs = scale_probs.clamp_min(1e-6)
            state_probs = state_probs / state_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
            state_loss = F.kl_div(state_probs.log(), state_target, reduction="batchmean")
            total = total + float(getattr(scale_cfg, "STATE_WEIGHT", 0.0)) * state_loss
            status["Loss/mstf_scale_state"] = state_loss.item()
            status["MSTF/state_small"] = state_probs[:, 0].mean().item()
            status["MSTF/state_mid"] = state_probs[:, 1].mean().item()
            status["MSTF/state_large"] = state_probs[:, 2].mean().item()
            status["MSTF/state_uncertain"] = state_probs[:, 3].mean().item()

        target_confidence = pred_dict.get("mstf_target_confidence", None)
        if target_confidence is not None:
            status["MSTF/target_confidence"] = target_confidence.detach().mean().item()
        residual_strength = pred_dict.get("mstf_residual_strength", None)
        if residual_strength is not None:
            status["MSTF/residual_strength"] = residual_strength.detach().mean().item()

        status["Loss/mstf_scale"] = total.item()
        status["MSTF/scale_small"] = target3[:, 0].mean().item()
        status["MSTF/scale_mid"] = target3[:, 1].mean().item()
        status["MSTF/scale_large"] = target3[:, 2].mean().item()
        return total, status

    @staticmethod
    def _soft_scale_target(scale_value, cfg):
        small_threshold = float(getattr(cfg, "SMALL_THRESHOLD", 0.16))
        large_threshold = float(getattr(cfg, "LARGE_THRESHOLD", 0.32))
        temperature = max(float(getattr(cfg, "TEMPERATURE", 0.04)), 1e-4)
        small = torch.sigmoid((small_threshold - scale_value) / temperature)
        large = torch.sigmoid((scale_value - large_threshold) / temperature)
        mid = (1.0 - small) * (1.0 - large)
        target = torch.stack([small, mid, large], dim=1).clamp_min(1e-6)
        return target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
