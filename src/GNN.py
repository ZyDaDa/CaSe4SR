import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

class GNN(MessagePassing):
    def __init__(self, out_channels: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.out_channels = out_channels

        self.linea_in = nn.Linear(out_channels, out_channels)
        self.linea_out = nn.Linear(out_channels, out_channels)

        self.rnn = torch.nn.GRUCell(out_channels*2, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=stdv)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        in_edge_weight = edge_weight[0]
        out_edge_weight = edge_weight[1]

            # m = torch.matmul(x, self.weight[i])
        m=x
        m_in = self.propagate(edge_index, x=m, edge_weight=in_edge_weight,
                            size=None)
        m_in = self.linea_in(m_in)


        m_out = self.propagate(torch.stack([edge_index[1],edge_index[0]],0), x=m, edge_weight=out_edge_weight,
                            size=None)
        m_out = self.linea_out(m_out)

        b = torch.concat([m_in,m_out],dim=-1)

        x = self.rnn(b, x)

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}')
