-- Author: Yonatan Belinkov
-- Last updated: July 1 2015
--
-- remove last element from a table 
-- based on nn.SelectTable

local RemoveTableLast, parent = torch.class('nn.RemoveTableLast', 'nn.Module')

function RemoveTableLast:__init()
   parent.__init(self)
   self.gradInput = {}
end

function RemoveTableLast:updateOutput(input)
   local output = {}
   for i = 1,#input-1 do 
     table.insert(output, input[i])
   end
   self.output = output
   return self.output
end

function RemoveTableLast:updateGradInput(input, gradOutput)
   local gradInput = {}
   for i = 1,#input do 
     if i == #input then
       table.insert(gradInput, input[i]:clone():zero())
     end
     table.insert(gradInput, gradOutput[i])
   end 
   self.gradInput = gradInput
   return self.gradInput
end

function RemoveTableLast:type(type)
   self.gradInput = {}
   self.output = {}
   return parent.type(self, type)
end