-- Author: Yonatan Belinkov
-- Last updated: October 17, 2015

-- require('mobdebug').start()
require 'torch'
require 'nn'
require 'rnn'
require 'RemoveTableLast'

----------------------------------
-- parse command line arguments
if not opt then
  print '==> parsing arguments' 
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-model', 'lstm', 'type of model: lstm | ???')
  -- TODO consider adding more layers
  cmd:option('-layers', 1, 'number of layers')
  cmd:option('-inputDim', 10, 'input dimensionality')
  -- TODO can this be different than inputDim? How to use as input to decoder?
  cmd:option('-encoderOutputDim', 10, 'encoder output dimensionality')
  cmd:option('-decoderOutputDim', 10, 'decoder output dimensionality')
  cmd:option('-dropout', 0.5, 'dropout rate')
  cmd:option('-loss', 'nll', 'loss function')
  cmd:option('-alphabetSize', 77, 'alphabet size')
  cmd:option('-numFeats', 11, 'number of morphological features')
  cmd:option('-batchSize', 1, 'batch size')
  cmd:text()
  opt = cmd:parse(arg or {})
end

alphabetSize = alphabetSize or opt.alphabetSize

----------------------------------
if opt.model == 'lstm' or opt.model == 'blstm' then 
  print('==> define ' .. opt.model .. ' model')
  
  -- encode lemma letters into vector
  lemmaEncoder = nn.Sequential()
  lookupTable = nn.LookupTable(alphabetSize, opt.inputDim)
  lemmaEncoder:add(lookupTable)
  lemmaEncoder:add(nn.SplitTable(1))
  lstm = nn.LSTM(opt.inputDim, opt.encoderOutputDim) 
  if opt.model == 'lstm' then
    lemmaEncoder:add(nn.Sequencer(lstm))
  elseif opt.model == 'blstm' then
    lemmaEncoder:add(nn.BiSequencer(lstm))
  end
  lemmaEncoder:add(nn.Sequencer(nn.Dropout(opt.dropout)))
  -- TODO mean pool instead of last one
  lemmaEncoder:add(nn.SelectTable(-1))
  -- dim reduction for bidirectional model
  if opt.model == 'blstm' then
    lemmaEncoder:add(nn.Linear(2*opt.encoderOutputDim, opt.encoderOutputDim))
  end

  -- encode morphological features into vector
  featsEncoder = nn.Sequential()
  -- TODO currently, feats and letters live in same vector space and have shared lookup table; consider moving feats to a different table
  featsLookupTable = lookupTable:clone('weight', 'gradWeight')
  featsEncoder:add(featsLookupTable)
  featsEncoder:add(nn.View(-1))
  -- TODO do we really need this projection layer? 
  featsEncoder:add(nn.Linear(opt.numFeats*opt.inputDim, opt.inputDim))

  -- read word letter vectors
  wordReader = nn.Sequential()
  wordLookupTable = lookupTable:clone('weight', 'gradWeight')
  wordReader:add(wordLookupTable)
  wordReader:add(nn.SplitTable(1))

  -- prepare table of lemma representation and word letter vectors
  -- input will be {lemma, word}, output will be {lemmaRepr, wordLetter1, wordLetter2, ...}
  lemmaWordPar = nn.ParallelTable()
  lemmaWordPar:add(lemmaEncoder):add(wordReader)
  lemmaWord = nn.Sequential()
  lemmaWord:add(lemmaWordPar)
  lemmaWord:add(nn.FlattenTable())

  -- concat morphological features rerpesentation to lemma representation and to all word letter vectors
  -- input will be {vec, table}, output will be {[vec tab[1]], [vec tab[2]], ...}, where [vec tab[i]] is concatenation of vec and i'th element in table
  concat = nn.Sequential()
  concat:add(nn.ZipTableOneToMany())
  concat:add(nn.Sequencer(nn.JoinTable(1)))
 
  -- decoder
  if opt.decoderAfterConcat then
    -- input to decoder is concatenation of feature vector representation of size inputDim (projected down from numFeats*inpuDim) and lemma encoding (outputDim)
    local decoderInputDim = opt.inputDim + opt.encoderOutputDim
    decoder = nn.Sequential()
    decoder:add(nn.Sequencer(nn.LSTM(decoderInputDim, opt.decoderOutputDim)))
    decoder:add(nn.Sequencer(nn.Dropout(opt.dropout)))
  else
    -- in this case we run decoder on the output form lemmaWord and only after that add feats representation
    local decoderInputDim = opt.inputDim
    decoder = nn.Sequential()
    decoder:add(nn.Sequencer(nn.LSTM(decoderInputDim, opt.decoderOutputDim)))
    lemmaWord:add(decoder)
  end

  -- model
  -- input will be {feats, {lemma, word}}
  par = nn.ParallelTable()
  par:add(featsEncoder)
  par:add(lemmaWord)
  model = nn.Sequential()
  model:add(par)
  model:add(concat)
  if opt.decoderAfterConcat then
    model:add(decoder)
  end
  model:add(nn.RemoveTableLast())

else
  error('unsupported model: ' .. opt.model)
end
      
print(model)   


------------------------------------------------------
if opt.loss == 'nll' then
  -- softmax
  softmax = nn.Sequential()
  if decoderAfterConcat then
    softmax:add(nn.Linear(opt.decoderOutputDim, alphabetSize))
  else
    softmax:add(nn.Linear(opt.decoderOutputDim+opt.encoderOutputDim, alphabetSize))
  end
  softmax:add(nn.LogSoftMax())
  model:add(nn.Sequencer(softmax))

  criterion = nn.ClassNLLCriterion()
  print '==> defined log-likelihood loss function:'
  print(criterion)
  seqCriterion = nn.SequencerCriterion(criterion)
  print 'defined sequencer criterion:'
  print(seqCriterion)
  
else
  error('unsupported loss: ' .. opt.loss)
end
  
-------------------- tests ---------------
uniTest = false
if uniTest then
  error("unitests not implemented")
end
  
  
