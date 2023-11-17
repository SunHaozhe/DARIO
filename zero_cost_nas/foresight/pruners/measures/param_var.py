# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import numpy as np
from . import measure
from ..p_utils import get_layer_metric_array

@measure('param_var', bn=False, mode='param')
def get_param_var_array(net, head_mask, alphas, inputs, targets, mode, split_data=1, loss_fn=None):
    device = inputs.device
    
    # Compute gradients with input of 1s 
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    net.zero_grad()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).to(device)
    output = net.forward_classification_feature(inputs, head_mask=head_mask, alphas=alphas)
    torch.sum(output).backward()
    opt.step()

    all_parameters = []
    for name, param in net.named_parameters():
        all_parameters.append(torch.flatten(param))
    
    all_parameters = torch.cat(all_parameters)
    res = [torch.var(all_parameters)]
    
    return res
