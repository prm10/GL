require 'torch'

local loader = {}
local class_name={[0]="正常",[1]="换炉"}

function loader.load_data()
  -- load
  local data = {}
  data.inputs = {}
  data.targets = {}
  data.targets_by_name = {}

  local f = torch.DiskFile("fscDataHWP.csv", "r")
  f:quiet()

  local line =  f:readString("*l")
  while line ~= '' do
      f1, f2, f3 = string.match(line, '([^,]+),([^,]+),([^,]+)')
      data.inputs[#data.inputs + 1] = tonumber(f2)
      data.targets[#data.targets + 1] = tonumber(f3)
      data.targets_by_name[#data.targets_by_name + 1] = class_name[tonumber(f3)]
      line = f:readString("*l")
  end

  data.inputs = torch.Tensor(data.inputs)
  data.targets = torch.Tensor(data.targets)

  print('--------------------------------')
  print('Loaded. Sizes:')
  print('inputs', data.inputs:size())
  print('targets', data.targets:size())
  print('--------------------------------')

  return data
end

function generateBatchData(batches,batch_size,length,delay)
  data.trainData={}
  data.trainLabel={}
  data.testData={}
  data.testLabel={}
  data.batches=batches
  data.batch_size=batch_size
  data.length=length
  data.delay=delay
  data.current_batch=1
  local n = torch.floor(#data.inputs*2.0/3)
  local step=torch.floor(n/batches/batch_size)
  local i=0
  while i<n-delay-length-step*batch_size do
    local trainData1=torch.Tensor({batch_size,length})
    local trainLabel1=torch.Tensor({batch_size,length})
    for j=1,batch_size do
      trainData1[j]=data.inputs:sub(i+1,i+length):t()
      trainLabel1[{j,{delay+1,length}}]=data.targets:sub(i+1,i+length-delay):t()
      trainLabel1[{j,{1,delay}}]=torch.zeros({1,delay})
      i=i+step
    end
    data.trainData[#data.trainData+1]=trainData1
    data.trainLabel[#data.trainLabel+1]=trainLabel1
  end
  data.testData[1]=data.inputs.sub(n+1,#data.inputs)
  local testLabel1=torch.Tensor({batch_size,length})
  testLabel1[{{},{delay+1,length}}]=data.targets.sub(n+1,#data.inputs-delay):t()
  testLabel1[{j,{1,delay}}]=torch.zeros({1,delay})
  data.testLabel[1]=testLabel1
end

function getTrainData()
  data.current_batch = (data.current_batch % data.batches) + 1
  return data.trainData[self.current_batch], data.trainLabel[self.current_batch]
end

function getTestData (args)
  return data.testData[self.current_batch], data.testLabel[self.current_batch]
end

return loader
