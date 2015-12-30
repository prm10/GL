require 'torch'

local loader = {}
loader.__index=loader

local class_name={[0]="正常",[1]="换炉"}

function loader.load_data()
  -- load
  local data = {}
  setmetatable(data, loader)
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
  print('data load done.')
  collectgarbage()
  return data
end

function loader:generateBatchData(batches,batch_size,length,delay)
  self.trainData={}
  self.trainLabel={}
  self.testData={}
  self.testLabel={}
  self.batches=batches
  self.batch_size=batch_size
  self.length=length
  self.delay=delay
  self.current_batch=0
  local n = torch.floor(self.inputs:size(1)*2/3)
  local step=torch.floor((n-delay-length)/batches/batch_size)
  local i=0
  while i<=n-delay-length-step*batch_size do
    local trainData1=torch.Tensor(batch_size,length):zero()
    local trainLabel1=torch.Tensor(batch_size,length):zero()
    for j=1,batch_size do
      trainData1[j]=self.inputs:narrow(1,i+1,length)
      trainLabel1[{j,{delay+1,length}}]=self.targets:narrow(1,i+1,length-delay)
      i=i+step
    end
    self.trainData[#self.trainData+1]=trainData1
    self.trainLabel[#self.trainLabel+1]=trainLabel1:add(1)
  end
  self.testData[1]=self.inputs:sub(n+1,self.inputs:size(1))
  local testLabel1=torch.Tensor(self.inputs:size(1)-n):zero()
  testLabel1[{{delay+1,testLabel1:size(1)}}]=self.targets:sub(n+1,self.targets:size(1)-delay)
  self.testLabel[1]=testLabel1:add(1)
  print('--------------------------------')
  print('trainset: '..#self.trainData)
  print('dataSize: ')
  print(self.trainData[1]:size())
  print('targetSize: ')
  print(self.trainLabel[1]:size())
  print('testset: '..#self.testData)
  print('--------------------------------')
  print('generate data done ')
  collectgarbage()
end

function loader:getTrainData()
  self.current_batch = (self.current_batch % self.batches) + 1
  return self.trainData[self.current_batch], self.trainLabel[self.current_batch]
end

function loader:getTestData (args)
  return self.testData[self.current_batch], self.testLabel[self.current_batch]
end

return loader
