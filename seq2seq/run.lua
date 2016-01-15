-- Author: Yonatan Belinkov
-- Last updated: December 6 2015

-- require('mobdebug').start()

if not opt then
  print '==> parsing arguments'
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  -- general
  cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
  cmd:option('-type', 'cuda', 'type: double | cuda')
  cmd:option('-device', 1, 'cuda device number (1-indexing)')
  -- data
  cmd:option('-data', 'turkish', 'data set: toy | turkish | arabic')
  cmd:option('-wordsTrainFile', 'turkish.ind.word', 'train file with words')
  cmd:option('-wordsTestFile', 'turkish.ind.word', 'test file with words')
  cmd:option('-lemmasTrainFile', 'turkish.ind.lemma', 'train file with lemmas')
  cmd:option('-lemmasTestFile', 'turkish.ind.lemma', 'test file with lemmas')
  cmd:option('-featsTrainFile', 'turkish.ind.feats', 'train file with morph features')
  cmd:option('-featsTestFile', 'turkish.ind.feats', 'test file with morph features')
  cmd:option('-alphabet', 'turkish.alphabet', 'alphabet file')
  -- model
  cmd:option('-model', 'blstm', 'model: lstm | blstm')
  cmd:option('-decoderAfterConcat', false, 'decode after (true) or before (false) concatenating morph features')
  -- TODO consider adding more layers
  cmd:option('-layers', 1, 'number of layers')
  cmd:option('-inputDim', 100, 'input dimensionality')
  cmd:option('-encoderOutputDim', 100, 'encoder output dimensionality')
  cmd:option('-decoderOutputDim', 100, 'decoder output dimensionality')
  cmd:option('-dropout', 0.2, 'dropout rate')
  cmd:option('-loss', 'nll', 'loss function')
  cmd:option('-alphabetSize', 77, 'alphabet size')
  cmd:option('-numFeats', 11, 'number of morphological features')
  -- train options
  cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
  cmd:option('-optimization', 'ADAGRAD', 'optimization method (SGD | ADAGRAD | RMSPROP)')
  cmd:option('-learningRate', 0.1, 'initial learning rate')
  -- TODO ?
  cmd:option('-maxGradNorm', 1, 'maximum value of grad norm')
  cmd:option('-maxGrad', 1000, 'maximuv value of each gradient')
  cmd:option('-batchSize', 1, 'batch size')
  cmd:option('-maxIter', 100, 'maximum number of iteration')
  -- test options
  cmd:option('-writePred', true, 'write prediction')
  cmd:option('-predFile', 'pred.txt', 'prediction output file')
  cmd:option('-testOutFile', 'test.txt', 'test file (with no empty lines)')
  -- TODO estimate better max decoding length from dev/train sets
  cmd:option('-maxDecodeLen', 3, 'maximum length of decoded string as times input length')
  cmd:text()
  opt = cmd:parse(arg or {})
  opt.rundir = cmd:string('experiment', opt, {dir=true})
  paths.mkdir(opt.rundir)
  cmd:log(opt.rundir .. '/log', opt)
end

-----------------------------------
torch.manualSeed(opt.seed)
math.randomseed(opt.seed)

if opt.type == 'cuda' then
  print('==> switching to CUDA')
  require 'cunn'
  torch.setdefaulttensortype('torch.FloatTensor')
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.device)
end

-----------------------------------
totalTime = sys.clock

dofile('data.lua')
dofile('model.lua')
dofile('train.lua')
dofile('test.lua')

test()
epoch = epoch or 1
while epoch <= opt.maxIter do
  train()
  test()
  epoch = epoch + 1
end
