-- Author: Yonatan Belinkov
-- Last updated: December 6 2015

-- require('mobdebug').start()
require 'torch'
require 'nn'
require 'xlua'
dofile('utils.lua')

---------------------------
if not opt then
  print '==> parsing arguments' 
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-save', 'results', 'subdirectory to save/log experiments in')  
  cmd:option('-writePred', true, 'write prediction')
  cmd:option('-predFile', 'pred.txt', 'prediction output file')  
  cmd:option('-testOutFile', 'test.txt', 'test file (with no empty lines)')
  -- TODO estimate better max decoding length from dev/train sets
  cmd:option('-maxDecodeLen', 1.5, 'maximum length of decoded string as times input length')
  cmd:text()
  opt = cmd:parse(arg or {})
end

-------------------------------------
print '==> defining test procedure'

function test()
  
  local time = sys.clock()
 
  -- TODO consider all this
  -- determine when to write representations and predictions
  local writeIter = false
  if opt.val and epoch == bestEpoch then
    -- if validating, write on every new best epoch
    writeIter = true 
  elseif not opt.val then
    -- if not validating, write on every epoch
    writeIter = true
  end        
  if writeIter then
    if opt.writeRepr then
      reprFile = torch.DiskFile(paths.concat(opt.save, opt.reprFile), 'w')
    end
    if opt.writePred then
      if epoch and epoch % 10 == 0 then
        predFile = torch.DiskFile(paths.concat(opt.save, opt.predFile .. '.epoch' .. epoch), 'w')
      else
        predFile = torch.DiskFile(paths.concat(opt.save, opt.predFile), 'w')
      end
      testOutFile = torch.DiskFile(paths.concat(opt.save, opt.testOutFile), 'w')
    end
  end
  
  -- necessary for dropout and for sequencer (different train/test behaviour)
  model:evaluate()
  -- TODO do something with remember and forget for LSTMs?
  recurRemember(decoder, 'eval')

  print('\n==> testing on test set:')
  local testSize = #wordsTestData
  for t = 1,testSize do 
    -- disp progress
    xlua.progress(t, testSize)

    -- get new sample
    local lemma, word, feats = lemmasTestData[t], wordsTestData[t], featsTestData[t]
    -- cuda input
    if opt.type == 'double' then 
      lemma, word, feats = lemma:double(), word:double(), feats:double() 
    elseif opt.type == 'cuda' then 
      lemma, word, feats = lemma:cuda(), word:cuda(), feats:cuda() 
    end
    
    -- get lemma and features representations
    local lemmaRepr = lemmaEncoder:forward(lemma)
    local featsRepr = featsEncoder:forward(feats)
    local lemmaFeatsRepr = torch.cat(featsRepr, lemmaRepr)
    -- print(lemmaRepr)
    -- print(featsRepr)
    -- print(lemmaFeatsRepr)


    -- decode
   
    -- forget previous inputs
    recurForget(decoder)

    local predIndices, predLetters = {}, {}
    local pred, bestProb, bestInd, predLetter
    -- predict first letter
    if decoderAfterConcat then
      -- print(lemmaFeatsRepr)
      -- print(decoder:forward({lemmaFeatsRepr}))
      pred = softmax:forward(decoder:forward({lemmaFeatsRepr})[1])
    else
      decoderOut = decoder:forward({lemmaRepr})[1]
      pred = softmax:forward(torch.cat(featsRepr, decoderOut))
    end
    -- TODO TMP check
    -- pred = softmax:forward(decoder:forward({lemmaRepr})[1])
    
    -- print(pred)
    bestProb, bestInd = torch.max(pred, 1)
    -- print('first bestInd: ' .. bestInd[1])
    -- print('first bestProb: ' .. bestProb[1])
    if bestInd[1] ~= alphabetSize then
      table.insert(predIndices, bestInd[1])
      table.insert(predLetters, alphabet[bestInd[1]])
    end
    maxTargetLen = torch.floor(lemma:size(1)*opt.maxDecodeLen)
    -- predict following letters
    i = 1
    while i < maxTargetLen and bestInd[1] ~= alphabetSize do
      local curLetterRepr = wordLookupTable:forward(bestInd):squeeze()
      if decoderAfterConcat then
        curLetterFeatsRepr = torch.cat(featsRepr, curLetterRepr)
        pred = softmax:forward(decoder:forward({curLetterFeatsRepr})[1])
      else
        curLetterOut = decoder:forward({curLetterRepr})[1]
	pred = softmax:forward(torch.cat(featsRepr, curLetterOut))
      end
      -- TODO TMP check
      -- pred = softmax:forward(decoder:forward({curLetterRepr})[1])
      -- print(pred)
      bestProb, bestInd = torch.max(pred, 1)
      -- print('bestInd: ' .. bestInd[1])
      -- print('bestProb: ' .. bestProb[1]) 
      if bestInd[1] ~= alphabetSize then
        table.insert(predIndices, bestInd[1])
	table.insert(predLetters, alphabet[bestInd[1]])
      end
      i = i + 1
    end
    if bestInd[1] == alphabetSize then print('==> Reached end token at index ' .. i) end

    -- print predictions
    for l, letter in ipairs(predLetters) do io.write(letter .. ' ') end
    io.write('\ninput size: ' .. lemma:size(1) .. ' decoded size: ' .. #predLetters .. '\n')
    -- write sentence representation
    if writeIter then
      if opt.writeRepr then
        local reprToWrite = opt.batchSize > 1 and repr[1] or repr 
        reprFile:writeString(vec2string(reprToWrite:double()) .. '\n')
      end
      if opt.writePred then
        --  writing letters
        predFile:writeString(stringx.join(' ', predLetters) .. '\n')
        local gold = {}
        for i = 1,word:size(1) do table.insert(gold, alphabet[word[i]]) end
        testOutFile:writeString(stringx.join(' ', gold) .. '\n')
      end
    end    
  end

   -- timing
  time = sys.clock() - time
  time = time / testSize
  print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

  if writeIter then
    if opt.writeRepr then
      io.write('==> wrote representations to: ' .. opt.reprFile .. '\n')
      reprFile:close()
    end
    if opt.writePred then
      io.write('==> wrote predictions to: ' .. opt.predFile .. '\n')
      predFile:close()
      testOutFile:close()
      if opt.eval then
        evaluate(opt.testOutFile, opt.predFile)
      end
    end
  end
  
  
end

  
