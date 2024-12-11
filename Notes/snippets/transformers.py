
### modeling_swinv2.py

# https://huggingface.co/microsoft/swinv2-large-patch4-window12-192-22k

self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

192*2^7=1536

self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)

# last_hidden_state

sequence_output = encoder_outputs[0]
sequence_output = self.layernorm(sequence_output)

最后一个hidden state
layer_outputs[0] -> Swinv2Stage -> Swinv2Layer -> Swinv2Attention -> Swinv2SelfAttention


# layer之间差异

class Swinv2Encoder
	dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
	stage_i = Swinv2Stage(drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],...)

class Swinv2Stage
	block_i = Swinv2Layer(drop_path_rate=drop_path[i], ...)

	for i, layer_module in enumerate(self.blocks):
        layer_head_mask = head_mask[i] if head_mask is not None else None

        layer_outputs = layer_module(
            hidden_states,
            input_dimensions,
            layer_head_mask,
            output_attentions,
        )

        hidden_states = layer_outputs[0]


# pooled_output

self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None
...
pooled_output = self.pooler(sequence_output.transpose(1, 2))
pooled_output = torch.flatten(pooled_output, 1)

用pooled_output做分类