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
    local features, guides = input[1],input[2]
    assert(features:dim() == guides:dim())
    local n_guides = nil
    local dtype = features:type()
    if guides:dim() == 3 then
       n_guides = guides:size()[1] 
    elseif guides:dim() == 4 then
       n_guides = guides:size()[2]
    end
    if self.mode == 'capture' then
        assert(guides:dim() == 4, 'style image and guide should have 4 dimensions')
            for i = 1, n_guides do
                self.agg[i] = nn.GramMatrixGuided():type(dtype)
                self.targets[i] = self.agg[i]:forward({features, guides[{{},{i},{},{}}]:expandAs(features):contiguous()}):clone()
            end
    elseif self.mode == 'loss' then
        self.loss = 0
        if guides:dim() == 3 then
            for i = 1, n_guides do
                local agg_out = self.agg[i]:forward({features, guides[{{i},{},{}}]:expandAs(features):contiguous()})
                self.loss = self.loss + self.strength * self.crit(agg_out, self.targets[i])
            end
        elseif guides:dim() == 4 then
            for i = 1, n_guides do
                local agg_out = self.agg[i]:forward({features, guides[{{},{i},{},{}}]:expandAs(features):contiguous()})
                self.loss = self.loss + self.strength * self.crit(agg_out, self.targets[i]:expandAs(agg_out):contiguous())
            end
        end
    end
    self.output = input
    return self.output
end


function StyleLossGuided:updateGradInput(input, gradOutput)
    local features, guides = input[1],input[2]
    local n_guides = nil
    if guides:dim() == 3 then
       n_guides = guides:size()[1] 
    elseif guides:dim() == 4 then
       n_guides = guides:size()[2]
    end
    if self.mode == 'capture' or self.mode == 'none' then
        self.gradInput = gradOutput
    elseif self.mode == 'loss' then
        self.gradInput = gradOutput
        if guides:dim() == 3 then
            for i = 1, n_guides do
                self.crit:backward(self.agg[i].output, self.targets[i])
                self.crit.gradInput:mul(self.strength)
                self.agg[i]:backward({features, guides[{{i},{},{}}]:expandAs(features):contiguous()}, self.crit.gradInput)
                self.gradInput[1]:add(self.agg[i].gradInput)
            end
        elseif guides:dim() == 4 then
            for i = 1, n_guides do
                self.crit:backward(self.agg[i].output, self.targets[i]:expandAs(self.agg[i].output))
                self.crit.gradInput:mul(self.strength)
                self.agg[i]:backward({features, guides[{{},{i},{},{}}]:expandAs(features):contiguous()}, self.crit.gradInput)
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
