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
import torch.nn as nn
import numpy as np
from . import measure

@measure('l0_norm', bn=False, mode='param')
def get_l0_norm_array(net, head_mask, alphas, inputs, targets, mode, split_data=1, loss_fn=None):
    device = inputs.device
    
    # Compute gradients with input of 1s 
    opt = torch.optim.Adam(net.parameters(), lr=0.1) #Large learning rate to ensure large changes in parameter values
    net.zero_grad()
    #net.double()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).to(device)
    output = net.forward_classification_feature(inputs, head_mask=head_mask, alphas=alphas)
    torch.sum(output).backward()
    opt.step()
    
    all_norms = []
    for name, param in net.named_parameters():
        all_norms.append(param.norm(0).reshape(1))
    
    all_norms = torch.cat(all_norms)
    res = [torch.mean(all_norms)]
    
    return res
