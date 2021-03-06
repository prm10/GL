require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
local data_loader = require "data_loader"
local LSTM = require 'LSTM'
local model_utils=require 'model_utils'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')

cmd:option('-seed',13,'seed')
cmd:option('-batches',10,'number of batch')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-seq_length',1000,'length of sequences to train on in parallel')
cmd:option('-delay',60,'time delay between targets and label')

cmd:option('-input_size',1,'size of input')
cmd:option('-rnn_size',20,'size of LSTM internal state')
cmd:option('-output_size',2,'size of output')

cmd:option('-max_epochs',100,'number of full passes through the training data')
cmd:option('-save_every',50,'save every 100 steps, overwriting the existing file')
cmd:option('-savefile','model_autosave','filename to autosave the model (protos) to, appended with the,param,string.t7')
cmd:option('-vocabfile','vocabfile.t7','filename of the string->int table')
cmd:option('-datafile','datafile.t7','filename of the serialized torch ByteTensor to load')

cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-loadfile','model_autosave','filename to load the model (protos)')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

-- preparation stuff:
torch.manualSeed(opt.seed)

-- 使用GPU
-- 需要传输到gpu中的数据（3类）：
-- lstm的c0, h0
-- data和label：x，y
-- lstm模型参数
local ok, cunn = pcall(require, 'cunn')
local ok2, cutorch = pcall(require, 'cutorch')
if not ok then print('package cunn not found!') end
if not ok2 then print('package cutorch not found!') end
if ok and ok2 and opt.gpuid >=0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
end

local loader=data_loader.load_data()
loader:generateBatchData(opt.batches,opt.batch_size,opt.seq_length,opt.delay)
--[[
x,y=loader:getTrainData()
gnuplot.plot({x[1],'+'},{y[1],'+'})

-- local ind=1
-- x1=x[{ind,{1,-opt.delay-1}}]
-- y1=y[{ind,{opt.delay+1,-1}}]
-- gnuplot.plot({torch.range(1,x1:size(1))[y1:eq(0)],x1[y1:eq(0)],'+'},
-- {torch.range(1,x1:size(1))[y1:eq(1)],x1[y1:eq(1)],'+'})
--]]

--[
-- define model prototypes for ONE timestep, then clone them
local create_model = require 'create_model'
-- print("load file"..opt.loadfile)
if opt.loadfile~=nil then
  print("load file"..opt.loadfile)
  protos = torch.load(opt.loadfile)
else
  print("create model!")
  protos = create_model(opt)
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.softmax)
if opt.loadfile==nil then
  params:uniform(-0.08, 0.08)
end
-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size)
local initstate_h = initstate_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()

if opt.gpuid >= 0 then
    initstate_c=initstate_c:cuda()
    initstate_h=initstate_h:cuda()
    dfinalstate_c=dfinalstate_c:cuda()
    dfinalstate_h=dfinalstate_h:cuda()
end
-- preprocessing helper function
function prepro(x,y)
    x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
    y = y:transpose(1,2):contiguous()
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    return x,y
end

-- do fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    x, y = loader:getTrainData()
    x,y = prepro(x,y)
    ------------------- forward pass -------------------
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{x[t]
        :resize(opt.batch_size,opt.input_size), lstm_c[t-1], lstm_h[t-1]})
        -- print(lstm_h[t])
        predictions[t] = clones.softmax[t]:forward(lstm_h[t])
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dx = {}                              -- d loss / d input embeddings
    local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                  -- output values of LSTM
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
        -- Two cases for dloss/dh_t:
        --   1. h_T is only used once, sent to the softmax (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the softmax and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == opt.seq_length then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)
        else
            dlstm_h[t]:add(clones.softmax[t]:backward(lstm_h[t], doutput_t))
        end

        -- backprop through LSTM timestep
        dx[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
            {x[t], lstm_c[t-1], lstm_h[t-1]},--x,c0,h0
            {dlstm_c[t], dlstm_h[t]}
        ))

        -- backprop through embeddings
        -- clones.embed[t]:backward(x[{{}, t}], dx[t])
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    -- initstate_c:copy(lstm_c[#lstm_c])
    -- initstate_h:copy(lstm_h[#lstm_h])

    -- clip gradient element-wise
    grad_params:clamp(-15, 15)

    return loss, grad_params
end

--[
-- optimization stuff
local losses = {}
-- local optim_state = {learningRate = 1e-1}
local optim_state = {learningRate=1e-4,momentum=0.9,weightDecay=0}
local iterations = opt.max_epochs * opt.batches
local time = 0
for i = 1, iterations do
    -- local _, loss = optim.adagrad(feval, params, optim_state)
    local timer = torch.Timer()
    local _, loss = optim.sgd(feval, params, optim_state)
    losses[#losses + 1] = loss[1]
    time = time + timer:time().real
    if i % opt.save_every == 0 then
        torch.save(opt.savefile, protos)
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f,gradnorm = %6.4e, time = %6.4f", i, loss[1], loss[1] / opt.seq_length,grad_params:norm(),time))
        time=0
    end
end

--]]

-- print(predictions[1]:exp()[{1,2}]/(predictions[1]:exp()[{1,1}]+predictions[1]:exp()[{1,2}]))
local index=1
local predict=torch.Tensor(opt.seq_length)
if opt.gpuid >= 0 then
    predict = predict:cuda()
end
for i=1,opt.seq_length do
  predict[i]=predictions[i]:exp()[{index,2}]/(predictions[i]:exp()[{index,1}]+predictions[i]:exp()[{index,2}])
end

gnuplot.figure()
gnuplot.plot(torch.Tensor(losses))
gnuplot.xlabel("迭代次数")
gnuplot.ylabel("误差")
gnuplot.figure()
gnuplot.plot({y[{{},{index}}]:float():add(-1)},{predict})
