require 'torch'
require 'nn'

require 'fast_neural_style.GramMatrixGuided'

local StyleLossGuided, parent = torch.class('nn.StyleLossGuided', 'nn.Module')


function StyleLossGuided:__init(strength, loss_type)
    parent.__init(self)
    self.strength = strength or 1.0
    self.loss = 0
    self.targets = {}
    self.agg = {} 
    self.agg_out = nil

    self.mode = 'none'
    loss_type = loss_type or 'L2'
    if loss_type == 'L2' then
        self.crit = nn.MSECriterion()
    elseif self.crit == 'SmoothL1' then
        self.crit = nn.SmoothL1Criterion()
    else
        error(string.format('invalid loss type "%s"', loss_type))
    end
end


function StyleLossGuided:updateOutput(input)
    local features, masks = input[1],input[2]
    local n_masks = nil
    local dtype = features:type()
    if masks:dim() == 3 then
       n_masks = masks:size()[1] 
    elseif masks:dim() == 4 then
       n_masks = masks:size()[2]
    end
    if self.mode == 'capture' then
        if masks:dim() == 3 then
            for i = 1, n_masks do
                self.agg[i] = nn.GramMatrixGuided():type(dtype)
                self.targets[i] = self.agg[i]:forward({features, masks[{{i},{},{}}]:expandAs(features):contiguous()}):clone()
            end
        elseif masks:dim() == 4 then
            for i = 1, n_masks do
                self.agg[i] = nn.GramMatrixGuided():type(dtype)
                self.targets[i] = self.agg[i]:forward({features, masks[{{},{i},{},{}}]:expandAs(features):contiguous()}):clone()
            end
        end
    elseif self.mode == 'loss' then
        self.loss = 0
        if masks:dim() == 3 then
            for i = 1, n_masks do
                local agg_out = self.agg[i]:forward({features, masks[{{i},{},{}}]:expandAs(features):contiguous()})
                self.loss = self.loss + self.strength * self.crit(agg_out, self.targets[i])
            end
        elseif masks:dim() == 4 then
            for i = 1, n_masks do
                local agg_out = self.agg[i]:forward({features, masks[{{},{i},{},{}}]:expandAs(features):contiguous()})
                self.loss = self.loss + self.strength * self.crit(agg_out, nn.utils.addSingletonDimension(self.targets[i]):expandAs(agg_out):contiguous())
            end
        end
    end
    self.output = input
    return self.output
end


function StyleLossGuided:updateGradInput(input, gradOutput)
    local features, masks = input[1],input[2]
    local n_masks = nil
    if masks:dim() == 3 then
       n_masks = masks:size()[1] 
    elseif masks:dim() == 4 then
       n_masks = masks:size()[2]
    end
    if self.mode == 'capture' or self.mode == 'none' then
        self.gradInput = gradOutput
    elseif self.mode == 'loss' then
        self.gradInput = gradOutput
        if masks:dim() == 3 then
            for i = 1, n_masks do
                self.crit:backward(self.agg[i].output, self.targets[i])
                self.crit.gradInput:mul(self.strength)
                self.agg[i]:backward({features, masks[{{i},{},{}}]:expandAs(features):contiguous()}, self.crit.gradInput)
                self.gradInput[1]:add(self.agg[i].gradInput)
            end
        elseif masks:dim() == 4 then
            for i = 1, n_masks do
                self.crit:backward(self.agg[i].output, nn.utils.addSingletonDimension(self.targets[i]):expandAs(self.agg[i].output))
                self.crit.gradInput:mul(self.strength)
                self.agg[i]:backward({features, masks[{{},{i},{},{}}]:expandAs(features):contiguous()}, self.crit.gradInput)
                self.gradInput[1]:add(self.agg[i].gradInput)
            end
        end
    end
    return self.gradInput
end


function StyleLossGuided:setMode(mode)
    if mode ~= 'capture' and mode ~= 'loss' and mode ~= 'none' then
        error(string.format('Invalid mode "%s"', mode))
    end
    self.mode = mode
end
