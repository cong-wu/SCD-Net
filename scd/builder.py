import torch
import torch.nn as nn

from .scd_encoder import PretrainingEncoder


# initilize weight
def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')


class SCD_Net(nn.Module):
    def __init__(self, args_encoder, dim=3072, K=65536, m=0.999, T=0.07):
        """
        args_encoder: model parameters encoder
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SCD_Net, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        print(" moco parameters", K, m, T)

        self.encoder_q = PretrainingEncoder(**args_encoder)
        self.encoder_k = PretrainingEncoder(**args_encoder)
        weights_init(self.encoder_q)
        weights_init(self.encoder_k)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        # domain level queues
        # temporal domain queue
        self.register_buffer("t_queue", torch.randn(dim, K))
        self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))

        # spatial domain queue
        self.register_buffer("s_queue", torch.randn(dim, K))
        self.s_queue = nn.functional.normalize(self.s_queue, dim=0)
        self.register_buffer("s_queue_ptr", torch.zeros(1, dtype=torch.long))

        # instance level queue
        self.register_buffer("i_queue", torch.randn(dim, K))
        self.i_queue = nn.functional.normalize(self.i_queue, dim=0)
        self.register_buffer("i_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, t_keys, s_keys, i_keys):
        N, C = t_keys.shape

        assert self.K % N == 0  # for simplicity

        t_ptr = int(self.t_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.t_queue[:, t_ptr:t_ptr + N] = t_keys.T
        t_ptr = (t_ptr + N) % self.K  # move pointer
        self.t_queue_ptr[0] = t_ptr

        s_ptr = int(self.s_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.s_queue[:, s_ptr:s_ptr + N] = s_keys.T
        s_ptr = (s_ptr + N) % self.K  # move pointer
        self.s_queue_ptr[0] = s_ptr

        i_ptr = int(self.i_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.i_queue[:, i_ptr:i_ptr + N] = i_keys.T
        i_ptr = (i_ptr + N) % self.K  # move pointer
        self.i_queue_ptr[0] = i_ptr

    def forward(self, q_input, k_input):
        """
        Input:
            time-majored domain input sequence: qc_input and kc_input
            space-majored domain input sequence: qp_input and kp_input
        Output:
            logits and targets
        """

        # compute temporal domain level, spatial domain level and instance level features
        qt, qs, qi = self.encoder_q(q_input)  # queries: NxC

        qt = nn.functional.normalize(qt, dim=1)
        qs = nn.functional.normalize(qs, dim=1)
        qi = nn.functional.normalize(qi, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            kt, ks, ki = self.encoder_k(k_input)  # keys: NxC

            kt = nn.functional.normalize(kt, dim=1)
            ks = nn.functional.normalize(ks, dim=1)
            ki = nn.functional.normalize(ki, dim=1)

        # interactive loss

        l_pos_ti = torch.einsum('nc,nc->n', [qt, ki]).unsqueeze(1)
        l_pos_si = torch.einsum('nc,nc->n', [qs, ki]).unsqueeze(1)
        l_pos_it = torch.einsum('nc,nc->n', [qi, kt]).unsqueeze(1)
        l_pos_is = torch.einsum('nc,nc->n', [qi, ks]).unsqueeze(1)

        l_neg_ti = torch.einsum('nc,ck->nk', [qt, self.i_queue.clone().detach()])
        l_neg_si = torch.einsum('nc,ck->nk', [qs, self.i_queue.clone().detach()])
        l_neg_it = torch.einsum('nc,ck->nk', [qi, self.t_queue.clone().detach()])
        l_neg_is = torch.einsum('nc,ck->nk', [qi, self.s_queue.clone().detach()])

        logits_ti = torch.cat([l_pos_ti, l_neg_ti], dim=1)
        logits_si = torch.cat([l_pos_si, l_neg_si], dim=1)
        logits_it = torch.cat([l_pos_it, l_neg_it], dim=1)
        logits_is = torch.cat([l_pos_is, l_neg_is], dim=1)

        logits_ti /= self.T
        logits_si /= self.T
        logits_it /= self.T
        logits_is /= self.T

        labels_ti = torch.zeros(logits_ti.shape[0], dtype=torch.long).cuda()
        labels_si = torch.zeros(logits_si.shape[0], dtype=torch.long).cuda()
        labels_it = torch.zeros(logits_it.shape[0], dtype=torch.long).cuda()
        labels_is = torch.zeros(logits_is.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(kt, ks, ki)

        return logits_ti, logits_si, logits_it, logits_is, \
               labels_ti, labels_si, labels_it, labels_is,