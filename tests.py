

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class NMTTEST(nn.Module):
    def __init__(self):
        print("testing")
    def test_gen_masks(self,batch_size,max_src_len,hidden_size):
        source_lengths = [20, 12, 9, 8, 5]
        enc_hiddens = torch.randn([batch_size,max_src_len,2*hidden_size])
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        
        print('done')


if __name__ == '__main__':
    batch_size =5
    max_src_len = 20
    hidden_size = 3
    nmttest = NMTTEST()
    nmttest.test_gen_masks(batch_size,max_src_len,hidden_size)

 ## Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t.
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
            