from torch import nn, optim, autograd
import pdb
import torch
from torchvision import datasets

class EBD(nn.Module):
    def __init__(self, flags):
      super(EBD, self).__init__()
      self.envs_num = len(flags.training_env)
      if self.flags.num_classes == 2:
      self.embedings = torch.nn.Embedding(self.envs_num, 1)
      else:
          self.embedings = torch.nn.Embedding(flags.envs_num, self.flags.num_classes)
      self.re_init()

    def re_init(self):
      self.embedings.weight.data.fill_(1.)

    def re_init_with_noise(self, noise_sd):
      if self.flags.num_classes == 2:
        rd = torch.normal(
          torch.Tensor([1.0] * self.envs_num),
          torch.Tensor([noise_sd] * self.envs_num))
        self.embedings.weight.data = rd.view(-1, 1).cuda()
      else:
        rd = torch.normal(
          torch.Tensor([1.0] * self.flags.envs_num * self.flags.num_classes),
          torch.Tensor([noise_sd] * self.flags.envs_num* self.flags.num_classes))
        self.embedings.weight.data = rd.view(-1, self.flags.num_classes).cuda()

    def forward(self, e):
      return self.embedings(e.long())