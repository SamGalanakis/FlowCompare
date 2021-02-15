import torch
import torch.nn.functional as F
from torch import nn

class ConditionalDenseNN(torch.nn.Module):

    def __init__(
            self,
            input_dim,
            context_dim,
            hidden_dims,
            param_dims=[1, 1],
            nonlinearity=torch.nn.ReLU(),
            residual_connections=True):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_multiplier = sum(param_dims)
        self.residual_connections = residual_connections

        # Calculate the indices on the output corresponding to each parameter
        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        # Create masked layers
        layers = [torch.nn.Linear(input_dim+context_dim, hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            layers.append(torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        layers.append(torch.nn.Linear(hidden_dims[-1], self.output_multiplier))
        # for layer in layers:
        #     nn.init.xavier_uniform_(layer.parameters(), gain=nn.init.calculate_gain('relu'))
        #Make last layer zeros in order to start as Identity
        with torch.no_grad():
            layers[-1].weight.copy_(torch.zeros_like(layers[-1].weight))
            layers[-1].bias.copy_(torch.zeros_like(layers[-1].bias))
        self.layers = torch.nn.ModuleList(layers)

        # Save the nonlinearity
        self.f = nonlinearity

    def forward(self, x, context):
        # We must be able to broadcast the size of the context over the input
        context = context.expand(x.size()[:-1]+(context.size(-1),))

        x = torch.cat([context, x], dim=-1)
        return self._forward(x)

    def _forward(self, x): 
        h = x
        residual = None
        for index, layer in enumerate(self.layers[:-1]):
            if ((index % 2) == 0) and index>1:
                h = layer(h)             
                h = self.f(residual+h)
            elif index ==0:
                h = self.f(layer(h))
            else:
                residual = h
                h = self.f(layer(h))
        h = self.layers[-1](h)

        # Shape the output, squeezing the parameter dimension if all ones
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_multiplier])

            if self.count_params == 1:
                return h

            else:
                return tuple([h[..., s] for s in self.param_slices])



class DenseNN(ConditionalDenseNN):
   

    def __init__(
            self,
            input_dim,
            hidden_dims,
            param_dims=[1, 1],
            nonlinearity=torch.nn.ReLU()
            ,residual_connections=True):
        super(DenseNN, self).__init__(
            input_dim,
            0,
            hidden_dims,
            param_dims=param_dims,
            nonlinearity=nonlinearity,
            residual_connections=residual_connections
        )

    def forward(self, x):
        return self._forward(x)