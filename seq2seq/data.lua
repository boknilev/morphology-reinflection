-- Author: Yonatan Belinkov
-- Last updated: December 2, 2015

-- require('mobdebug').start()
require 'torch'
stringx = require 'pl.stringx'


-- parse command line arguments
if not opt then
  print '==> parsing arguments' 
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-inputDim', 10, 'input dimensionality')
  cmd:option('-wordsTrainFile', 'turkish.ind.word', 'train file with words')
  cmd:option('-wordsTestFile', 'turkish.ind.word', 'test file with words')
  cmd:option('-lemmasTrainFile', 'turkish.ind.lemma', 'train file with lemmas')
  cmd:option('-lemmasTestFile', 'turkish.ind.lemma', 'test file with lemmas')
  cmd:option('-featsTrainFile', 'turkish.ind.feats', 'train file with morph features')
  cmd:option('-featsTestFile', 'turkish.ind.feats', 'test file with morph features')
  cmd:option('-alphabet', 'turkish.alphabet', 'alphabet file')
  cmd:text()
  opt = cmd:parse(arg or {})
end

------------------------------------------
dataDir = "../data"

print '==> Loading alphabet'
alphabetFile = paths.concat(dataDir, opt.alphabet)
alphabet = {}
for line in io.lines(alphabetFile) do
  local letter = stringx.strip(line)
  table.insert(alphabet, letter)
end 
-- alphabet size includes end token
alphabetSize = #alphabet + 1
print('==> Alphabet size: ' .. alphabetSize)


print '==> Loading data'
wordsTrainFile = paths.concat(dataDir, opt.wordsTrainFile)
wordsTestFile = paths.concat(dataDir, opt.wordsTestFile)
lemmasTrainFile = paths.concat(dataDir, opt.lemmasTrainFile)
lemmasTestFile = paths.concat(dataDir, opt.lemmasTestFile)
featsTrainFile = paths.concat(dataDir, opt.featsTrainFile)
featsTestFile = paths.concat(dataDir, opt.featsTestFile)

function loadData(dataFile, reverseSource, endToken)
  local dataSet = {}
  for line in io.lines(dataFile) do
    if stringx.strip(line) ~= "" then
      local splt = stringx.split(line)
      local vec = endToken and torch.DoubleTensor(#splt+1):fill(0) or torch.DoubleTensor(#splt):fill(0)   
      for i = 1,#splt do
        idx = tonumber(splt[i])
	assert(idx <= #alphabet, "letter index exceeds alphabet size")
        vec[i] = idx
      end
      -- reverse source data
      if reverseSource then
	vec = vec:index(1, torch.range(vec:size(1),1,-1):long())
      end
      -- add end symbol to target data
      if endToken then
        vec[#splt+1] = alphabetSize
      end
      -- print(vec)
      table.insert(dataSet, vec)
    end
  end
  return dataSet
end

print('==> loading training data') 
wordsTrainData = loadData(wordsTrainFile, false, true)
lemmasTrainData = loadData(lemmasTrainFile, true, false)
featsTrainData = loadData(featsTrainFile)
assert(#wordsTrainData == #lemmasTrainData and #wordsTrainData == #featsTrainData, "number of train words, lemmas, and feats must be equal")
print('==> Number of training examples: ' .. #wordsTrainData)
print('==> loading testing data')
wordsTestData = loadData(wordsTestFile, false, true)
lemmasTestData = loadData(lemmasTestFile, true, false)
featsTestData = loadData(featsTestFile)
assert(#wordsTestData == #lemmasTestData and #wordsTestData == #featsTestData, "number of test words, lemmas, and feats must be equal")
print('==> Number of testing examples: ' .. #wordsTestData)


---[[ 
if true then
  for d, dataSet in ipairs({wordsTrainData, wordsTestData, lemmasTrainData, lemmasTestData}) do
    for i, dataPoint in ipairs(dataSet) do
      if i == 11 then
        break
      end
      print('==> ' .. i)
      -- print(dataPoint)
      for l=1,dataPoint:size(1) do
        local letter = alphabet[dataPoint[l]]
	if letter then io.write(letter .. " ") else io.write('_END_' .. " ") end
      end  
      print '\n=========================='
    end
  end
end

--]]
