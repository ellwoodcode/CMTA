"""
Binary classification model wrapping CMTA encoders/bridge/decoders.

Final head nn.Linear(2*d,1); BCEWithLogitsLoss; flag --detach_align detaches p and g for L1 alignment.
"""

import torch
import torch.nn as nn


from cmta_models.cmta.network import CMTA


class CMTABinary(nn.Module):
    """
    Binary classification model based on CMTA encoders and decoders.

    Args:
        omic_sizes (list of int): input dims for genomic embeddings.
        fusion (str): fusion type passed to CMTA ('concat' or 'bilinear').
        model_size (str): model size for CMTA ('small' or 'large').
        detach_align (bool): if True, detach p and g for L1 alignment loss.
    """

    def __init__(
        self,
        omic_sizes,
        fusion="concat",
        model_size="small",
        detach_align=True,
        path_size=None,
    ):
        super().__init__()
        # backbone CMTA for feature extraction
        self.cmta = CMTA(
            omic_sizes=omic_sizes,
            n_classes=2,
            fusion=fusion,
            model_size=model_size,
            path_size=path_size,
        )
        # override classification head
        d = self.cmta.size_dict["pathomics"][model_size][-1]
        self.head = nn.Linear(2 * d, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.detach_align = detach_align

    def forward(self, x_path, x_embed):
        """
        Args:
            x_path (Tensor): pathomics features [batch, num_patches, feat_dim].
            x_embed (Tensor or list of Tensors): genomic embeddings [batch, embed_dim] or list.
        Returns:
            logits (Tensor): raw logits [batch].
            align_loss (Tensor): L1 alignment loss between p and g.
            p (Tensor): pathomics representation [batch, d].
            g (Tensor): genomics representation [batch, d].
        """
        # unwrap single-sample batch dims: CMTA expects unbatched inputs
        if x_path.dim() == 3 and x_path.size(0) == 1:
            _x_path = x_path.squeeze(0)
        else:
            _x_path = x_path
        # prepare genomic inputs, unwrap batch dim if present
        if isinstance(x_embed, (list, tuple)):
            x_omic = [e.squeeze(0) if (e.dim() == 2 and e.size(0) == 1) else e for e in x_embed]
        else:
            x_omic = [x_embed.squeeze(0) if (x_embed.dim() == 2 and x_embed.size(0) == 1) else x_embed]

        # run CMTA backbone
        _, _, p_enc, p_dec, g_enc, g_dec = self.cmta(
            x_path=_x_path,
            **{f"x_omic{i+1}": x_omic[i] for i in range(len(x_omic))},
        )
        # representations
        p = (p_enc + p_dec) / 2
        g = (g_enc + g_dec) / 2
        # alignment loss
        if self.detach_align:
            align_loss = self.l1_loss(p.detach(), g.detach())
        else:
            align_loss = self.l1_loss(p, g)
        # classification head
        logits = self.head(torch.cat([p, g], dim=1)).squeeze(1)
        return logits, align_loss, p, g

    def loss(self, logits, labels, align_loss):
        """
        Compute total loss = BCEWithLogitsLoss + L1 alignment loss.
        """
        labels = labels.view(-1).float()
        bce = self.bce_loss(logits, labels)
        return bce + align_loss