import torch
A = torch.randn([12, 9, 64])
B = torch.randn([12, 9, 64])
Ar = A.repeat(1, 1, 9).view(12, 81, 64)
Br = B.repeat(1, 9, 1)
C = torch.cat((Ar, Br), dim=2)
D = torch.cat([A.unsqueeze(2).expand(-1, -1, 9, -1),
               B.unsqueeze(1).expand(-1, 9, -1, -1)], dim=-1).view(12, 81, 128)
print ((C-D).abs().max().item())

##
# The outter most [] is the 0th dimension
# A_{i,j,k}
A = torch.randn([1, 2, 3])
A
A.view(3, 2, 1)   # the inner most [] (corresponding to the last dim) only has 1 element
A.repeat(1, 2, 5)  # repeat the 0th dimension 1 times, the 1th dimension 2 times
