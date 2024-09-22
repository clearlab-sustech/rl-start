from typing import List, Optional

import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: List[int],
        activate_func: Optional[str] = 'relu',
        normalize_output: Optional[bool] = False,
        print_info: Optional[bool] = False,
        name: Optional[str] = 'MLP',
    ) -> None:
        '''A clase for Multi-Layer Perceptron (MLP).

        Params
        ------
        input_size: int
            size of input data.
        output_size: int
            size of output features.
        hidden_size: List[int]
            a list containing the size of each hidden layer.
        activate_func: Optional[str], default = 'relu'
            the name of the activation function.
        normalize_output: Optional[bool] = False
            normalize the output of the MLP if True.
        print_info: Optional[bool], default = False
            print the information of each layer if True.
        name: Optional[str], default = 'MLP'
            give the current MLP object a name
        '''
        super(MLP, self).__init__()

        assert activate_func.lower() in ['elu', 'selu', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        self.activate_func = get_activation_func(activate_func)

        self.normalize_output = normalize_output
        
        self.name = name

        layers = []
        if len(hidden_size) == 1:
            layers.append(nn.Linear(input_size, hidden_size[0]))
            layers.append(self.activate_func)
            layers.append(nn.Linear(hidden_size[0], output_size))
        else:
            layers.append(nn.Linear(input_size, hidden_size[0]))
            layers.append(self.activate_func)
            for hidden_idx in range(len(hidden_size)):
                if hidden_idx < len(hidden_size) - 1:
                    layers.append(nn.Linear(hidden_size[hidden_idx], hidden_size[hidden_idx+1]))
                    layers.append(self.activate_func)
            layers.append(nn.Linear(hidden_size[-1], output_size))
        
        self.mlp = nn.Sequential(*layers)

        if print_info:
            if normalize_output:
                print(f"The {self.name} Structure (MLP): {self.mlp}, output_normalization = True")
            else:
                print(f"The {self.name} Structure (MLP): {self.mlp}, output_normalization = False")

    def forward(self, x) -> torch.Tensor:
        output = self.mlp(x)
        if self.normalize_output:
            output = torch.nn.functional.normalize(output, p=2, dim=-1)
        return output
    
    def __str__(self) -> str:
        return f"The {self.name} Structure (MLP): {self.mlp}, output_normalization = True" \
            if self.normalize_output \
            else f"The {self.name} Structure (MLP): {self.mlp}, output_normalization = False"


def get_activation_func(activation_name: str):
    if activation_name.lower() == "elu":
        return nn.ELU()
    elif activation_name.lower() == "selu":
        return nn.SELU()
    elif activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "leakyrelu":
        return nn.LeakyReLU()
    elif activation_name.lower() == "tanh":
        return nn.Tanh()
    elif activation_name.lower() == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


if __name__ == '__main__':
    mlp1 = MLP(10, 20, [30], name='MLP_with_1_hidden_layers')
    mlp2 = MLP(10, 20, [30, 28], name='MLP_with_2_hidden_layers')
    mlp3 = MLP(10, 20, [30, 28, 27], name='MLP_with_3_hidden_layers')
    print(mlp1)
    print(mlp2)
    print(mlp3)

    input_tensor = torch.randn(50, 10)
    print(mlp1.forward(input_tensor))
    print(mlp1.forward(input_tensor).shape)