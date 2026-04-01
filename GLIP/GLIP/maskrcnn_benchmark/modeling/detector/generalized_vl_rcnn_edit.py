class GeneralizedVLRCNN(nn.Module):
    def __init__(self, cfg):
        super(GeneralizedVLRCNN, self).__init__()
        # ... 原有初始化代码（省略） ...

        if abs(cfg.IMPROMPT.gvl) == 1:
            self.reference_image_tensor = None
            self.reference_gt = None
            self.reference_length = 0
            self.reference_length_make_sure_same_dataset = 0
            self.reference_image_tensor_make_sure_same_dataset = []
            self.history_reference_image_tensor = [0 for x in range(0, 80)]
            # NEW: 初始化三个新分支的存储数组（和原有bbox分支对齐）
            self.reference_image_tensor_center = []  # 中心分支prompt
            self.reference_image_tensor_mask = []  # 实例掩码分支prompt
            self.reference_image_tensor_contour = []  # 轮廓分支prompt
            self.history_reference_image_tensor_center = [0 for x in range(0, 80)]
            self.history_reference_image_tensor_mask = [0 for x in range(0, 80)]
            self.history_reference_image_tensor_contour = [0 for x in range(0, 80)]

        # ... 其余原有初始化代码（省略） ...

    def forward(self, images, targets=None, captions=None, positive_map=None, greenlight_map=None, reference_map=None,
                idx=0):
        # ... 原有 forward 开头逻辑（省略） ...

        if abs(self.cfg.IMPROMPT.gvl) == 1:
            if self.cfg.IMPROMPT.input_way == 'input_image_itself':
                if self.training:
                    reference_map = targets
                    if self.reference_image_tensor is None or self.cfg.INPUT.FIX_RES:
                        # 重置所有分支的存储数组
                        self.reference_image_tensor = []  # 原bbox分支
                        self.reference_image_tensor_center = []  # NEW: 中心分支
                        self.reference_image_tensor_mask = []  # NEW: 掩码分支
                        self.reference_image_tensor_contour = []  # NEW: 轮廓分支

                        import copy
                        imprompt = copy.deepcopy(images)
                        from ..evaluation_utils import img_preprocess
                        bbs = targets[0].bbox
                        self.reference_length = bbs.shape[0]
                        labels = targets[0].extra_fields['labels']

                        for box_id in range(bbs.shape[0]):
                            # ========== 1. 原有bbox分支prompt收集（保留） ==========
                            mask_of_reference = np.zeros((images.tensors[0].shape[-2], images.tensors[0].shape[-1]))
                            THIS_BOX = bbs[box_id, :]
                            x1, y1, x2, y2 = THIS_BOX.cpu()
                            mask_of_reference[int(np.round(y1)):int(np.round(y2)),
                            int(np.round(x1)):int(np.round(x2))] = 1
                            out_reference_image_bbox = [
                                img_preprocess((None, [imprompt.tensors[0, :, :, :]], [mask_of_reference]), blur=3,
                                               bg_fac=0.1).numpy()[0]
                            ]
                            out_tensor_bbox = torch.from_numpy(out_reference_image_bbox[0]).unsqueeze(0).cuda().type(
                                torch.float32)
                            self.reference_image_tensor.append(out_tensor_bbox)

                            # ========== 2. NEW: 中心（center）分支prompt收集 ==========
                            mask_center = np.zeros((images.tensors[0].shape[-2], images.tensors[0].shape[-1]))
                            centerx = (x1 + x2) / 2
                            centery = (y1 + y2) / 2
                            # 中心区域：以目标中心为核心，取10x10的区域（可配置）
                            k = 10
                            center_x1 = max(0, int(centerx - k))
                            center_y1 = max(0, int(centery - k))
                            center_x2 = min(images.tensors[0].shape[-1], int(centerx + k))
                            center_y2 = min(images.tensors[0].shape[-2], int(centery + k))
                            mask_center[center_y1:center_y2, center_x1:center_x2] = 1
                            out_reference_image_center = [
                                img_preprocess((None, [imprompt.tensors[0, :, :, :]], [mask_center]), blur=3,
                                               bg_fac=0.2, whitelize=True).numpy()[0]
                            ]
                            out_tensor_center = torch.from_numpy(out_reference_image_center[0]).unsqueeze(
                                0).cuda().type(torch.float32)
                            self.reference_image_tensor_center.append(out_tensor_center)

                            # ========== 3. NEW: 实例掩码（mask）分支prompt收集 ==========
                            mask_np = np.zeros((images.tensors[0].shape[-2], images.tensors[0].shape[-1]),
                                               dtype=np.uint8)
                            try:
                                # 从target中获取实例掩码
                                mask_polygons = reference_map[0].extra_fields['masks'].polygons
                                polygon = mask_polygons[box_id]
                                pts = polygon.polygons[0]
                                ptss = pts.view((-1, 2)).numpy().reshape((-1, 1, 2)).astype(np.int32)
                                cv2.fillPoly(mask_np, [ptss], color=1)
                            except:
                                # 兜底：掩码不存在时用bbox区域
                                mask_np[int(np.round(y1)):int(np.round(y2)), int(np.round(x1)):int(np.round(x2))] = 1
                            out_reference_image_mask = [
                                img_preprocess((None, [imprompt.tensors[0, :, :, :]], [mask_np]), blur=3,
                                               bg_fac=0.1).numpy()[0]
                            ]
                            out_tensor_mask = torch.from_numpy(out_reference_image_mask[0]).unsqueeze(0).cuda().type(
                                torch.float32)
                            self.reference_image_tensor_mask.append(out_tensor_mask)

                            # ========== 4. NEW: 轮廓（contour）分支prompt收集 ==========
                            mask_contour = np.zeros((images.tensors[0].shape[-2], images.tensors[0].shape[-1]),
                                                    dtype=np.uint8)
                            try:
                                # 从掩码提取轮廓
                                contours, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(mask_contour, contours, -1, 1, 3)  # 轮廓宽度3
                            except:
                                # 兜底：用bbox边缘作为轮廓
                                mask_contour[int(np.round(y1)):int(np.round(y1)) + 3,
                                int(np.round(x1)):int(np.round(x2))] = 1
                                mask_contour[int(np.round(y2)) - 3:int(np.round(y2)),
                                int(np.round(x1)):int(np.round(x2))] = 1
                                mask_contour[int(np.round(y1)):int(np.round(y2)),
                                int(np.round(x1)):int(np.round(x1)) + 3] = 1
                                mask_contour[int(np.round(y1)):int(np.round(y2)),
                                int(np.round(x2)) - 3:int(np.round(x2))] = 1
                            out_reference_image_contour = [
                                img_preprocess((None, [imprompt.tensors[0, :, :, :]], [mask_contour]), blur=3,
                                               bg_fac=0.1).numpy()[0]
                            ]
                            out_tensor_contour = torch.from_numpy(out_reference_image_contour[0]).unsqueeze(
                                0).cuda().type(torch.float32)
                            self.reference_image_tensor_contour.append(out_tensor_contour)

                    # 随机选择一个prompt ID（和原有逻辑对齐）
                    selected_reference_ID = random.randint(0, self.reference_length - 1)

                    # ========== 5. NEW: 收集选中的四个分支prompt ==========
                    # 原bbox分支
                    output_to_feed_to_backbone_bbox = self.reference_image_tensor[selected_reference_ID]
                    # 新增三个分支
                    output_to_feed_to_backbone_center = self.reference_image_tensor_center[selected_reference_ID]
                    output_to_feed_to_backbone_mask = self.reference_image_tensor_mask[selected_reference_ID]
                    output_to_feed_to_backbone_contour = self.reference_image_tensor_contour[selected_reference_ID]

                    # 组合成数组（输入网络后端的核心数组）
                    output_to_feed_to_backbone_all = [
                        output_to_feed_to_backbone_bbox,
                        output_to_feed_to_backbone_center,
                        output_to_feed_to_backbone_mask,
                        output_to_feed_to_backbone_contour
                    ]

                else:
                    # TEST阶段逻辑：和TRAIN对齐，收集四个分支的prompt数组
                    # ... 原有test阶段bbox分支逻辑（省略） ...
                    # NEW: 同步收集center/mask/contour分支，最终生成output_to_feed_to_backbone_all数组
                    pass

                # ========== 6. NEW: 多分支prompt输入backbone，生成embedding ==========
                if self.cfg.IMPROMPT.shot_num == 1 or self.training:
                    # 每个分支单独过backbone，得到各自的embedding
                    reference_embeding_bbox = self.backbone(output_to_feed_to_backbone_bbox)
                    reference_embeding_center = self.backbone(output_to_feed_to_backbone_center)
                    reference_embeding_mask = self.backbone(output_to_feed_to_backbone_mask)
                    reference_embeding_contour = self.backbone(output_to_feed_to_backbone_contour)

                    # 多分支embedding融合（可选方式：平均/拼接/加权，这里默认平均，可配置）
                    fusion_mode = self.cfg.IMPROMPT.fusion_mode if hasattr(self.cfg.IMPROMPT,
                                                                           'fusion_mode') else 'average'
                    reference_embeding = []
                    for level in range(5):  # 对应backbone的5个stage特征
                        feat_bbox = reference_embeding_bbox[level]
                        feat_center = reference_embeding_center[level]
                        feat_mask = reference_embeding_mask[level]
                        feat_contour = reference_embeding_contour[level]

                        if fusion_mode == 'average':
                            fused_feat = (feat_bbox + feat_center + feat_mask + feat_contour) / 4.0
                        elif fusion_mode == 'concat':
                            # 拼接后用1x1卷积降维回256通道
                            fused_feat = torch.cat([feat_bbox, feat_center, feat_mask, feat_contour], dim=1)
                            fused_feat = nn.Conv2d(256 * 4, 256, kernel_size=1).to(fused_feat.device)(fused_feat)
                        elif fusion_mode == 'weighted':
                            # 可学习加权（需提前初始化weight参数）
                            weight = self.branch_weight.to(feat_bbox.device)  # [4,]的可学习权重
                            fused_feat = weight[0] * feat_bbox + weight[1] * feat_center + weight[2] * feat_mask + \
                                         weight[3] * feat_contour
                        else:
                            fused_feat = feat_bbox  # 兜底用原bbox分支

                        reference_embeding.append(fused_feat)
                else:
                    # shot_num>1时的逻辑：遍历每个shot，收集多分支prompt
                    reference_embeding = []
                    for shot_i in range(output_to_feed_to_backbone_all[0].shape[0]):
                        # 每个shot的四个分支prompt
                        bbox_shot = output_to_feed_to_backbone_all[0][shot_i:shot_i + 1, :, :, :]
                        center_shot = output_to_feed_to_backbone_all[1][shot_i:shot_i + 1, :, :, :]
                        mask_shot = output_to_feed_to_backbone_all[2][shot_i:shot_i + 1, :, :, :]
                        contour_shot = output_to_feed_to_backbone_all[3][shot_i:shot_i + 1, :, :, :]

                        # 每个分支过backbone
                        emb_bbox = self.backbone(bbox_shot)
                        emb_center = self.backbone(center_shot)
                        emb_mask = self.backbone(mask_shot)
                        emb_contour = self.backbone(contour_shot)

                        # 融合后加入列表
                        fused_emb = []
                        for level in range(5):
                            fused_feat = (emb_bbox[level] + emb_center[level] + emb_mask[level] + emb_contour[
                                level]) / 4.0
                            fused_emb.append(fused_feat)
                        reference_embeding.append(fused_emb)

            # ========== 7. 原有map_module逻辑（兼容多分支embedding） ==========
            if self.cfg.IMPROMPT.map_module == 'mlp':
                visual_embeding_flatten = []
                for ii, feat_per_level in enumerate(reference_embeding[-1:]):
                    feat_per_level = self.adaptivepool_of_stages[ii](feat_per_level)
                    bs, c, h, w = feat_per_level.shape
                    feat = permute_and_flatten(feat_per_level, bs, 1, c, h, w)
                    visual_embeding_flatten.append(feat)
                visual_embeding_flatten = cat(visual_embeding_flatten, dim=1)

            # ========== 8. 多分支prompt的embedding传入网络后端 ==========
            # 原有embed_768_to_lang逻辑（兼容融合后的embedding）
            if self.cfg.IMPROMPT.gvl == -1:
                if self.cfg.IMPROMPT.embed_768_to_lang >= 0:
                    # ... 原有embed逻辑（省略） ...
                    # 最终将融合后的embedding传入language_dict_features，进入RPN/ROI Heads
                    pass

        # ========== 9. 网络后端输入（RPN） ==========
        proposals, proposal_losses, fused_visual_features = self.rpn(
            images, visual_features, targets, language_dict_features,
            positive_map, captions, swint_feature_c4, reference_feature
        )
        # ... 其余原有逻辑（省略） ...