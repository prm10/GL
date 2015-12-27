require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
local data_loader = require "data_loader"


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')

cmd:option('-seed',13,'seed')
cmd:option('-batches',50,'number of batch')
cmd:option('-batch_size',20,'number of sequences to train on in parallel')
cmd:option('-length',1000,'length of sequences to train on in parallel')
cmd:option('-delay',60,'time delay between targets and label')

cmd:text()

-- parse input params
local opt = cmd:parse(arg)

-- preparation stuff:
torch.manualSeed(opt.seed)

local data=data_loader.load_data()
data.generateBatchData(opt.batches,opt.batch_size,opt.length,opt.delay)
x,y=data.getTrainData()
gnuplot.plot(y:eq(0),x[y:eq(0)])
