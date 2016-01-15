-- Author: Yonatan Belinkov
-- Last updated: October 17, 2015

-- require('mobdebug').start()
require 'torch'
require 'xlua'
require 'optim'

------------------------------------
-- parse command line arguments
if not opt then
  print '==> parsing arguments' 
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
  cmd:option('-optimization', 'SGD', 'optimization method (SGD | ADAGRAD | RMSPROP)')
  cmd:option('-learningRate', 0.1, 'initial learning rate')
  -- TODO implement
  cmd:option('-maxGradNorm', 1, 'maximum value of grad norm')
  cmd:option('-maxGrad', 1000, 'maximuv value of each gradient')
  cmd:option('-batchSize', 1, 'batch size')
  cmd:option('-maxIter', 7, 'maximum number of iteration')
  cmd:text()
  opt = cmd:parse(arg or {})
end

---------------------------
-- CUDA
if opt.type == 'cuda' then
  model:cuda()
  criterion:cuda()
  seqCriterion:cuda()
  if featsLookupTable then featsLookupTable:share(lookupTable, 'weight', 'gradWeight') end
  if wordLookupTable then wordLookupTable:share(lookupTable, 'weight', 'gradWeight') end

end

-----------------------------
-- log files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the model
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

-----------------------------------
print '==> configuring optimizer'

if opt.optimization == 'SGD' then
  optimState = {
    learningRate = opt.learningRate,
    -- weightDecay = opt.weightDecay,
    -- momentum = opt.momentum,
    -- learningRateDecay = 1e-7
  }
  optimMethod = optim.sgd
elseif opt.optimization == 'RMSPROP' then 
  optimState = {
    -- learningRate = opt.learningRate,
    -- weightDecay = opt.weightDecay,
    -- momentum = opt.momentum,
    -- learningRateDecay = 1e-7
  }
  optimMethod = optim.rmsprop
elseif opt.optimization == 'ADAGRAD' then 
  optimState = {
    learningRate = opt.learningRate,
    -- weightDecay = opt.weightDecay,
    -- momentum = opt.momentum,
    -- learningRateDecay = 1e-7
  }
  optimMethod = optim.adagrad
else
  error('unknown optimization method: ' .. opt.optimization)
end

--------------------------------------
print '==> defining training procedure'

function train()
    
  local time = sys.clock()
  
   -- set model to training mode (for modules that differ in training and testing, like Dropout)  
  model:training()
  local trainSize = #lemmasTrainData
  local shuffle = torch.randperm(trainSize)
  
  -- keep errors
  local f_s = {}
    
   -- do one epoch
  print('\n==> doing epoch on training data:')
  print('==> online epoch # ' .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  for t = 1,trainSize, opt.batchSize do 
    -- display progress
    xlua.progress(t, trainSize)
    
    -- create mini batch
    local inputs, targets = {}, {}
    for i = t,math.min(t+opt.batchSize-1, trainSize) do 
      local lemma = lemmasTrainData[shuffle[i]]
      local word = wordsTrainData[shuffle[i]]
      local feats = featsTrainData[shuffle[i]]
      if opt.type == 'double' then 
        lemma, word, feats = lemma:double(), word:double(), feats:double()
      elseif opt.type == 'cuda' then 
	lemma, word, feats = lemma:cuda(), word:cuda(), feats:cuda()
      end
      table.insert(inputs, {feats, {lemma, word}})
      table.insert(targets, word)
    end

    -- create closure to evaluate f(x) and df/dx
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end
      
      -- reset gradients
      gradParameters:zero()
      
      -- f is the average of all criterions
      local f = 0
      
      -- evaluate function for complete minibatch
      for i = 1,#inputs do
        -- estimate f
        
	-- print(inputs[i])
        local output = model:forward(inputs[i])
        local err = seqCriterion:forward(output, targets[i])
        f = f + err
	-- print(err)
        
        -- estimate df/dw
        local df_do = seqCriterion:backward(output, targets[i])
        model:backward(inputs[i], df_do)
	-- print(df_do)
      end
      
      -- normalize gradients
      gradParameters:div(#inputs)
      f = f/#inputs
      table.insert(f_s, f)
      
      -- clip gradients
      gradNorm = gradParameters:norm()
      gradParameters:clamp(-opt.maxGrad, opt.maxGrad)            
      if gradParameters:norm() ~= gradNorm then
        print('==> gradient norm changed after clipping')
      end
      
      -- return f and df/dX
      return f,gradParameters
    end
    
    -- optimize on current minibatch
    optimMethod(feval, parameters, optimState)
  end
  
  
  time = sys.clock() - time
  time = time / trainSize
  print('==> time to learn 1 sample = ' .. (time*1000) .. 'ms')
  print('==> average error = ' .. torch.mean(torch.Tensor{f_s}))
  trainLogger:add{['% average error (train set)'] = torch.mean(torch.Tensor{f_s}) }
  
end

