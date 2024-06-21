# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid


class DNDeformableDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        operation_order: tuple = ("sparse_self_attn", "norm", "ffn", "norm"),
        num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,
    ):
        super(DNDeformableDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=operation_order,
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
        self.sample_ratio = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    def forward(
        self,
        start_idx,
        end_idx,
        query,
        key,
        value,
        reference_points,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        attn_map=None,
        **kwargs,
    ):


        output = query
        valid_tokens_nums_all_imgs = (~query_key_padding_mask).int().sum(dim=1)
        _, _, d_model = query.shape
        _, _, num_lvl, _ = reference_points.shape

        for lid in range(start_idx, end_idx):
            if attn_map is not None:
                valid_enc_token_num =  (valid_tokens_nums_all_imgs * self.sample_ratio[lid] ).int() + 1
                batch_token_num = max(valid_enc_token_num)
                topk_enc_token_indice = attn_map.topk(batch_token_num, dim=1)[1] # (bs, batch_token_num)

                query_pos_ = query_pos.gather(dim=1, index=topk_enc_token_indice.unsqueeze(dim=2).repeat(1, 1, d_model))
                reference_points_ = reference_points.gather(dim=1, index=topk_enc_token_indice.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, num_lvl, 2)) # (x, y) for ref points
                query_ = output.gather(dim=1, index=topk_enc_token_indice.unsqueeze(dim=2).repeat(1, 1, d_model))
            else:
                query_ = output
                query_pos_ = query_pos
                reference_points_ = reference_points
            layer = self.layers[lid]
            output_query, _, _ = layer(
                query_,
                key,
                value,
                query_pos=query_pos_,
                attn_masks=attn_masks,
                reference_points=reference_points_,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            if attn_map is not None:
                outputs=[]
                for img_id, (num, idx) in enumerate(zip(valid_enc_token_num, topk_enc_token_indice)):
                    valid_idx = idx[:num]
                    # src[0]: (ori_num_token, 256)
                    # idx: (valid_num,) -> (valid_num, 256)
                    # sparse_memory[0]: (max_token_num, 256)
                    # src[0][index[i,j], j] = sparse_memory[0][i,j]
                    outputs.append(output[img_id].scatter(dim=0, index=valid_idx.unsqueeze(1).repeat(1, d_model), src=output_query[img_id][:num]))
                output = torch.stack(outputs)
            else:
                output = output_query

        assert self.post_norm_layer is  None
        return output


class DNDeformableDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
    ):
        super(DNDeformableDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate

        self.query_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        start_idx,
        end_idx,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,  # num_queries, 4
        valid_ratios=None,
        **kwargs,
    ):
        output = query
        bs, num_queries, _ = output.size()
        # reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 4

        intermediate = []
        intermediate_reference_points = []
        locations = []
        weights = []
        for layer_idx in range(start_idx, end_idx):
            layer = self.layers[layer_idx]
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if layer_idx != 0 else 1
            query_pos = pos_scale * raw_query_pos

            output, sampling_locations, attn_weights= layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )
            
            locations.append(sampling_locations)
            weights.append(attn_weights)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return intermediate, intermediate_reference_points, locations, weights

        return output, reference_points


class DNDeformableDetrTransformer(nn.Module):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        as_two_stage=False,
        num_feature_levels=4,
        two_stage_num_proposals=300,
    ):
        super(DNDeformableDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_outpout_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        input_label_query,
        input_box_query,
        attn_masks,
        **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        enc_reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory = feat_flatten

        bs, _, c = memory.shape
        assert not self.as_two_stage
        dec_reference_points = input_box_query.sigmoid()
        target = input_label_query
        init_reference_out = dec_reference_points
        inter_states = []
        inter_references = []

        combos = (
            (0, 0, 0, 1),
            (0, 1, 1, 2),
            (1, 3, 2, 3),
            (3, 6, 3, 6)
        )
        attn_map = None
        for enc_start_idx, enc_end_idx, dec_start_idx, dec_end_idx in combos:
            memory = self.encoder(
                start_idx=enc_start_idx,
                end_idx=enc_end_idx,
                query=memory,
                key=memory,
                value=memory,
                reference_points=enc_reference_points,  # bs, num_token, num_level, 2
                query_pos=lvl_pos_embed_flatten,
                # attn_masks=attn_masks,
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                attn_map=attn_map,
                **kwargs,
            )

            # decoder
            inter_states_, inter_references_, sampling_locations, attn_weights = self.decoder(
                start_idx=dec_start_idx,
                end_idx=dec_end_idx,
                query=target,  # bs, num_queries, embed_dims
                key=memory,  # bs, num_tokens, embed_dims
                value=memory,  # bs, num_tokens, embed_dims
                query_pos=None,
                attn_masks=attn_masks,
                key_padding_mask=mask_flatten,  # bs, num_tokens
                reference_points=dec_reference_points,  # num_queries, 4
                spatial_shapes=spatial_shapes,  # nlvl, 2
                level_start_index=level_start_index,  # nlvl
                valid_ratios=valid_ratios,  # bs, nlvl, 2
                **kwargs,
            )
            inter_states = inter_states + inter_states_
            inter_references = inter_references + inter_references_
            target = inter_states_[-1]
            dec_reference_points = inter_references_[-1]
            
            sampling_locations = torch.stack(sampling_locations, dim=1)
            attn_weights = torch.stack(attn_weights, dim=1)
            attn_map = attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations, attn_weights).sum(dim=(1,2))
        inter_states = torch.stack(inter_states)
        inter_references = torch.stack(inter_references)
        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out, None, None

def attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations, attention_weights):
    # sampling_locations: [N, n_layers, Len_q, n_heads, n_levels, n_points, 2]
    # attention_weights: [N, n_layers, Len_q, n_heads, n_levels, n_points]
    N, n_layers, _, n_heads, *_ = sampling_locations.shape
    sampling_locations = sampling_locations.permute(0, 1, 3, 2, 5, 4, 6).flatten(0, 2).flatten(1, 2)
    # [N * n_layers * n_heads, Len_q * n_points, n_levels, 2]
    attention_weights = attention_weights.permute(0, 1, 3, 2, 5, 4).flatten(0, 2).flatten(1, 2)
    # [N * n_layers * n_heads, Len_q * n_points, n_levels]

    rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1) # hw -> wh (xy)
    col_row_float = sampling_locations * rev_spatial_shapes

    col_row_ll = col_row_float.floor().to(torch.int64)
    zero = torch.zeros(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    one = torch.ones(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    col_row_lh = col_row_ll + torch.stack([zero, one], dim=-1)
    col_row_hl = col_row_ll + torch.stack([one, zero], dim=-1)
    col_row_hh = col_row_ll + 1

    margin_ll = (col_row_float - col_row_ll).prod(dim=-1)
    margin_lh = -(col_row_float - col_row_lh).prod(dim=-1)
    margin_hl = -(col_row_float - col_row_hl).prod(dim=-1)
    margin_hh = (col_row_float - col_row_hh).prod(dim=-1)

    flat_grid_shape = (attention_weights.shape[0], int(torch.sum(spatial_shapes[..., 0] * spatial_shapes[..., 1])))
    flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device)

    zipped = [(col_row_ll, margin_hh), (col_row_lh, margin_hl), (col_row_hl, margin_lh), (col_row_hh, margin_ll)]
    for col_row, margin in zipped:
        valid_mask = torch.logical_and(
            torch.logical_and(col_row[..., 0] >= 0, col_row[..., 0] < rev_spatial_shapes[..., 0]),
            torch.logical_and(col_row[..., 1] >= 0, col_row[..., 1] < rev_spatial_shapes[..., 1]),
        )
        idx = col_row[..., 1] * spatial_shapes[..., 1] + col_row[..., 0] + level_start_index
        idx = (idx * valid_mask).flatten(1, 2)
        weights = (attention_weights * valid_mask * margin).flatten(1)
        flat_grid.scatter_add_(1, idx, weights)

    return flat_grid.reshape(N, n_layers, n_heads, -1)