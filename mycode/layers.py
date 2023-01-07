import torch
from torch import nn
from torch.nn.parameter import Parameter

class AdditiveAttentionLayer(nn.Module):
    def __init__(self, input_shape, latent_dim=32 ,kernel_regularizer = None,
            **kwargs):
        super(AdditiveAttentionLayer, self).__init__()
      
        self.latent_dim = latent_dim
        self.kernel_regularizer = kernel_regularizer

        in_seq_shape = input_shape[0]
        out_shape = input_shape[1]
        # Create a trainable weight variable for this layer.
        self.Wa = Parameter(torch.Tensor(size= (in_seq_shape[-1] , self.latent_dim)))
        self.Ua = Parameter(torch.Tensor(size= (out_shape[1], self.latent_dim)))
        self.Va = Parameter(torch.Tensor(size= (self.latent_dim, 1)))

        self.apply(init_weights)

    def init_weights(self, m):
        torch.nn.init.uniform_(m.weight)
      
    def forward(self, in_seq, out_vec):
        out_vec_shape = out_vec.size()

        ## reshape input sequence from(batchsize,timesteps,features) to (batchsize*timesteps,features)
        in_seq_shape = in_seq.size()
        in_seq_reshape = in_seq.view((in_seq_shape[0]*in_seq_shape[1],-1))

        ## Compute
        W_as = torch.dot(in_seq_reshape, self.Wa)
        ##
        out = torch.dot(out_vec, self.Ua)

        out = (out.repeat(size=in_seq_shape[1])).view((in_seq_shape[0]*in_seq_shape[1],-1))

        energy = torch.dot(torch.tanh(W_as+out), self.Va).view((in_seq_shape[0],-1))

        ## prob have shape(batchsize,timesteps)
        prob = torch.softmax(energy)
        # print('Shape of prob:')
        # print(K.int_shape(prob))
        
        contxt_vec =  torch.sum(in_seq * torch.unsqueeze(prob, dim=-1), dim=1)

        print('Shape of context vector:')
        print(contxt_vec.size())

        return contxt_vec

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])


class SelfAttentionLayer(nn.Module):

    def __init__(self, input_shape, latent_dim=32,
                    kernel_regularizer = None,
                    **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
      
        self.latent_dim = latent_dim
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # self.bias_regularizer = regularizers.get(bias_regularizer)

        timesteps = input_shape[1]
        h_dim = input_shape[2]
        # Create a trainable weight variable for this layer.
        self.WQ = Parameter(torch.Tensor(size= (h_dim , self.latent_dim)))
        self.WK = Parameter(torch.Tensor(size= (h_dim , self.latent_dim)))

#         self.Va = self.add_weight(name='Va',
#                                       shape=(latent_dim, 1),
#                                       initializer='uniform',
#                                       trainable=True)
        self.apply(init_weights)

    def init_weights(self, m):
        torch.nn.init.uniform_(m.weight)

    def forward(self, inputs):
        in_seq = inputs

        ## reshape input sequence from(batchsize,timesteps,features) to (batchsize*timesteps,features)
        in_seq_shape = in_seq.size()
#        in_seq_reshape = K.reshape(in_seq,(in_seq_shape[0]*in_seq_shape[1],-1))

        ## Compute
        query = torch.dot(in_seq,self.WQ)
        ##
        key = torch.dot(in_seq,self.WK)
        # print(K.int_shape(key))

        energy = torch.bmm(query, key.view((0,2,1)))/self.latent_dim

        #   #### Apply masking prob here
        #   masking_prob = np.ones((K.int_shape(key)[1],K.int_shape(key)[1]))
        #   masking_prob = np.tril(masking_prob, k=0)

        #   ## apply masking to energy

        #   energy = energy * masking_prob

        ## prob have shape(batchsize,timesteps)
        prob = torch.softmax(energy, dim=-1)
        print('Shape of prob:')
        print(prob.size()))

#         out = K.batch_dot(prob,in_seq)
        return prob

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])
