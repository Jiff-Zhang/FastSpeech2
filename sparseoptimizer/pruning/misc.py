
def get_max_prune_rate(in_channel):
    if in_channel <= 8:
        return 0.0
    elif in_channel <= 16:
        return 1 / 2
    elif in_channel <= 32:
        return 3 / 4
    elif in_channel <= 64:
        return 7 / 8
    else:
        return 1.0


def generate_prune_dict(model, model_name, sparsity, verbose=False):
    prune_dict = {}
    set_up_info = {'cgb':{},'dtype':{}}
    for name, parameter in model.named_parameters():
        # DeBERTa
        if 'deberta' in model_name.lower():
            # FFN
            if ('intermediate.dense.weight' in name or 'output.dense.weight' in name) \
                and ('attention' not in name):
                prune_dict[name] = sparsity
            # Attention
            # Optional: prune all kind of attentions(i.e. pos_proj, pos_q_proj)
            # if 'attention.self.in_proj.weight' in name or 'attention.self.pos_proj.weight' in name \
            #     or 'attention.self.pos_q_proj.weight' in name or 'attention.output.dense.weight' in name:
            if 'attention.self.in_proj.weight' in name or 'attention.output.dense.weight' in name:
                prune_dict[name] = sparsity
        # Bert
        elif 'bert' in model_name.lower():
            # FFN
            if ('intermediate.dense.weight' in name or 'output.dense.weight' in name) \
                and ('attention' not in name):
                prune_dict[name] = sparsity
            # Attention
            if 'attention.self.query.weight' in name or 'attention.self.key.weight' in name or \
                'attention.self.value.weight' in name or 'attention.output.dense.weight' in name:
                prune_dict[name] = sparsity
        
        elif 'fastspeech' in model_name.lower():
            if len(parameter.shape) < 2:
                if verbose:
                    print('skip dim=1', name, parameter.shape)
                continue
            elif 'layer' in name and 'weight' in name:
                if verbose:
                    print('prune en/decoder', name, parameter.shape)
                prune_dict[name] = sparsity
                set_up_info['cgb'][name] = 64
                set_up_info['dtype'][name] = 'int8'
            elif 'adaptor' in name and 'weight' in name and parameter.shape[0]!=1:
                if verbose:
                    print('prune adaptor', name, parameter.shape)
                prune_dict[name] = sparsity
                set_up_info['cgb'][name] = 64
                set_up_info['dtype'][name] = 'int8'
            elif 'postnet' in name and 'weight' in name:
                if verbose:
                    print('prune postnet', name, parameter.shape)
                prune_dict[name] = sparsity
                set_up_info['cgb'][name] = 64
                set_up_info['dtype'][name] = 'int8'
            
        
        elif 'conformer' in model_name.lower():
            num_encoders = len(model.module.encoder)
            if len(parameter.shape) < 2:
                if verbose:
                    print('skip because shape dim <2', name)
                continue
            # skip the first and the last encoder
            if 'encoder.0' in name or 'encoder.1' in name in name or f'encoder.{num_encoders - 1}' in name or name.endswith('bias'):
                if verbose:
                    print('skip because of first, second, last, bias:', name, parameter.shape)
                continue
            if len(parameter.shape) == 3 and parameter.shape[1] == 1:
                if verbose:
                    print('skip depthwise conv', name, parameter.shape)
                continue

            if 'linear.weight' in name:
                if verbose:
                    print('prune', name, parameter.shape)
                prune_dict[name] = sparsity
                set_up_info['cgb'][name] = 64
                set_up_info['dtype'][name] = 'int8'
            if 'conv.weight' in name:
                if verbose:
                    print('prune conv', name, parameter.shape)
                prune_dict[name] = sparsity
                set_up_info['cgb'][name] = 32
                set_up_info['dtype'][name] = 'bfloat16'

        elif 't5' in model_name.lower():
            # if model_config:
            #     num_encoder_layers = getattr(model_config, 'num_layers', 0)
            #     num_decoder_layers = getattr(model_config, 'num_decoder_layers', 0)
            
            '''1st & last layer of encoder'''
            # TODO: clarify this strategy
            # if 'encoder.block.0' in name:
            #     continue
            # TODO: clarify this strategy
            # if model_config and f'encoder.block.{num_encoder_layers - 1}' in name:
            #     continue

            '''1st & last layer of decoder'''
            # TODO: clarify this strategy
            # if 'decoder.block.0' in name:
            #     continue
            # TODO: clarify this strategy
            # if model_config and f'decoder.block.{num_decoder_layers - 1}' in name:
            #     continue

            # FFN
            if 'DenseReluDense' in name:
                sub_module = ['wi', 'wo']
                for mod_name in sub_module:
                    if f'DenseReluDense.{mod_name}.weight' in name:
                        prune_dict[name] = sparsity
            # Attention
            if 'SelfAttention' in name or 'EncDecAttention' in name:
                sub_module = ['q', 'k', 'v', 'o']
                for mod_name in sub_module:
                    if f'Attention.{mod_name}.weight' in name:
                        prune_dict[name] = sparsity  # if mod_name != 'o' else 0.5 * sparsity
        # Other cases
        else:
            if 'weight' in name and len(parameter.shape)>=2:
                if len(parameter.shape)==4:
                    if list(parameter.shape)[2:] == [1, 1]:
                        max_prune_rate = get_max_prune_rate(list(parameter.shape)[1])
                    else:
                        max_prune_rate = 1.0
                else:
                    max_prune_rate = 0.75
            prune_dict[name] = min(sparsity, max_prune_rate)
    print(set_up_info)
    return prune_dict, set_up_info
