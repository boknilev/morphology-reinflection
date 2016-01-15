-- Author: Yonatan Belinkov
-- Last updated: July 27 2015


function vec2string(vec, delim)
  d = delim or ' '
  return stringx.join(d, vec:totable())
end


-- set Sequencer modules to remember
function recurRemember(model, mode)
  for i=1,#model.modules do
    if torch.isTypeOf(model.modules[i], nn.Sequencer) then
      model.modules[i]:remember(mode)
    end
  end
end

-- old version
-- set Sequencer modules to remember
function recurRememberOld(model)
  for i=1,#model.modules do
    if torch.isTypeOf(model.modules[i], nn.Sequencer) then
      model.modules[i]:remember()
    end
  end
end

-- set Sequencer modules to forget
function recurForget(model)
  for i=1,#model.modules do
    if torch.isTypeOf(model.modules[i], nn.Sequencer) then
      model.modules[i]:forget()
    end
  end
end

-- TODO functions below this are probably obsolete
-- get mini batch
function getBatch(trainSourceData, trainTargetData, shuffle, t, opt)
  local inputs = {}
  local targets = {}
  local maxInputLength, maxTargetLength = 0, 0
  shuffle = shuffle or torch.range(1,#trainSourceData) -- in minibatch training do not shuffle
  for i = t,math.min(t+opt.batchSize-1, #trainSourceData) do
    local input = trainSourceData[shuffle[i]]
    local target = trainTargetData[shuffle[i]]
    if opt.type == 'double' then input = input:double()
    elseif opt.type == 'cuda' then input = input:cuda() end
    table.insert(inputs, input)
    table.insert(targets, target) 
    if input:size(1) > maxInputLength then
    	maxInputLength = input:size(1)
    end
    if target:size(1) > maxTargetLength then
    	maxTargetLength = target:size(1)
    end
  end
  return inputs, targets, maxInputLength, maxTargetLength
end

-- put batch in matrices for efficient processing
function getBatchMatrices(inputs, targets, opt, maxInputLength, maxTargetLength)
  -- all sentences in a minibatch should be of the same size, but if not, pad with ones (index for unknown word)
  -- inputMat[i][j] is the j'th word in the i'th input sentence
  if opt.type == 'double' then
    inputMat = torch.ones(opt.batchSize, maxInputLength):double()
    targetMat = torch.ones(opt.batchSize, maxTargetLength):double()
  elseif opt.type == 'cuda' then
    inputMat = torch.ones(opt.batchSize, maxInputLength):cuda()
    targetMat = torch.ones(opt.batchSize, maxTargetLength):cuda()      
  end
  for i = 1,#inputs do
    if inputs[i]:size(1) ~= maxInputLength then
      error('==> Error: size of input ' .. i .. ' (' .. inputs[i]:size(1) .. ') != max input batch length (' .. maxInputLength .. ')\n')
    elseif targets[i]:size(1) ~= maxTargetLength then
      error('==> Error: size of target ' .. i .. ' (' .. targets[i]:size(1) .. ') != max target batch length (' .. maxTargetLength .. ')\n')
    end
    inputMat[i] = inputs[i]
    targetMat[i] = targets[i]
  end
  -- targetTable[i][j] is i'th word in the j'th sentence
  targetTable = nn.SplitTable(2):forward(targetMat)
  
  return inputMat, targetMat, targetTable
end



