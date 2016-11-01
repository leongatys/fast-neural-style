require 'torch'
require 'nn'

local Gram, parent = torch.class('nn.GramMatrixGuided', 'nn.Module')


--[[
A layer to compute the Gram Matrix of inputs.

Input:
A table with entries:
- features: A tensor of shape (N, C, H, W) or (C, H, W) giving features for
  either a single image or a minibatch of images.
- guides: A tensor of shape (N, C, H, W) or (C, H, W) with 
  spatial guidance maps that are multiplied with the feature maps.

Output:
- gram: A tensor of shape (N, C, C) or (C, C) giving spatially guided Gram matrix for input
--]]


function Gram:__init(normalize)
  parent.__init(self)
  self.normalize = normalize or true
  self.buffer = torch.Tensor()
end


function Gram:updateOutput(input)
  local C, H, W
  if input[1]:dim() == 3 then
    C, H, W = input[1]:size(1), input[1]:size(2), input[1]:size(3)
    local x_flat = input[1]:view(C, H * W)
    local g_flat = input[2]:view(C, H * W)
    local gx_flat = torch.cmul(x_flat,g_flat)
    self.output:resize(C, C)
    self.output:mm(gx_flat, gx_flat:t())
  elseif input[1]:dim() == 4 then
    local N = input[1]:size(1)
    C, H, W = input[1]:size(2), input[1]:size(3), input[1]:size(4)
    self.output:resize(N, C, C)
    local x_flat = input[1]:view(N, C, H * W)
    local g_flat = input[2]:view(N, C, H * W)
    local gx_flat = torch.cmul(x_flat,g_flat)
    self.output:resize(N, C, C)
    self.output:bmm(gx_flat, gx_flat:transpose(2, 3))
  end
  if self.normalize then
    -- print('in gram forward; dividing by ', C * H * W)
    self.output:div(C * H * W)
  end
  return self.output
end


function Gram:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input[1]):zero()
  local C, H, W
  if input[1]:dim() == 3 then
    C, H, W = input[1]:size(1), input[1]:size(2), input[1]:size(3)
    local x_flat = input[1]:view(C, H * W)
    local g_flat = input[2]:view(C, H * W)
    local gx_flat = torch.cmul(x_flat,g_flat)
    self.buffer:resize(C, H * W)
    self.buffer:mm(gradOutput, gx_flat)
    self.buffer:addmm(gradOutput:t(), gx_flat)
    self.buffer:cmul(g_flat)
    self.gradInput = self.buffer:view(C, H, W)
  elseif input[1]:dim() == 4 then
    local N = input[1]:size(1)
    C, H, W = input[1]:size(2), input[1]:size(3), input[1]:size(4)
    local x_flat = input[1]:view(N, C, H * W)
    local g_flat = input[2]:view(N, C, H * W)
    local gx_flat = torch.cmul(x_flat,g_flat)
    self.buffer:resize(N, C, H * W)
    self.buffer:bmm(gradOutput, gx_flat)
    self.buffer:baddbmm(gradOutput:transpose(2, 3), gx_flat)
    self.buffer:cmul(g_flat)
    self.gradInput = self.buffer:view(N, C, H, W)
  end
  if self.normalize then
    self.buffer:div(C * H * W)
  end
  assert(self.gradInput:isContiguous())
  return self.gradInput
end

