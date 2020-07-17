import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from configuration_bert import BertConfig
from bert_file_utils import WEIGHTS_NAME, hf_bucket_url, cached_path


# Bert Model
class BertPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and load pretrained models."""

    def __init__(self, config, *model_args, **model_kwargs):
        super(BertPreTrainedModel, self).__init__()
        self.config = config
        self.base_model_prefix = "bert"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        if attention_mask.dim() == 3:  # attention_mask: bs x from_len_seq x to_len_seq
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:  # attention_mask: bs x ls
            if self.config.is_decoder:
                batch_size, len_seq = input_shape
                seq_ids = torch.arange(len_seq, device=device, dtype=attention_mask.dtype)
                causal_mask = seq_ids.repeat(batch_size, len_seq, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        # encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * 1e-4

        return encoder_extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        # head mask: [num_heads] or [num_hidden_layers x num_heads]
        if head_mask is not None:
            head_mask = self._get_head_mask(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)  # ?
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask
        # num_hidden_layers x batch x num_heads x len_seq x leq_seq
        # or: a list of None with length num_hidden_layers

    def _get_head_mask(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:  # [num_heads]
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            # num_hidden_layers x batch x num_heads x len_seq x leq_seq
        elif head_mask.dim() == 2:  # [num_hidden_layers x num_heads
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, "Invalid head mask"
        return head_mask

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **model_kwargs):
        config = model_kwargs.pop("config", None)
        cache_dir = model_kwargs.pop("cache_dir", './')
        use_cdn = model_kwargs.pop("use_cdn", True)
        force_download = model_kwargs.pop("force_download", False)
        state_dict = model_kwargs.pop("state_dict", None)

        # set config
        if not isinstance(config, BertConfig):
            config = BertConfig()
            config.config_path = pretrained_model_name_or_path

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)

            else:
                archive_file = hf_bucket_url(pretrained_model_name_or_path, WEIGHTS_NAME, use_cdn=use_cdn)

            try:  # download or load existing files
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download, )
            except EnvironmentError:
                raise EnvironmentError("Invalid url.")

            model = cls(config, *model_args, **model_kwargs)

            if state_dict is None:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")

            missing_keys = []
            unexpected_keys = []
            error_msgs = []

            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=""):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys,
                                             error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            start_prefix = ""
            model_to_load = model
            # has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
            load(model_to_load, prefix=start_prefix)

            model.eval()
            return model
