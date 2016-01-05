require 'torch'
require 'nn'
require 'nngraph'
local LSTM = require 'LSTM'
local model_utils=require 'model_utils'
local create_model = require 'create_model'

--------------------------------------------------------------
-- SETTINGS

-- function that numerically checks gradient of the loss:
-- f is the scalar-valued function
-- g returns the true gradient (assumes input to f is a 1d tensor)
-- returns difference, true gradient, and estimated gradient
local function checkgrad(f, g, x, eps)
  -- compute true gradient
  f(x)
  local grad = g(x)

  -- compute numeric approximations to gradient
  local eps = eps or 1e-7
  local grad_est = torch.DoubleTensor(grad:size())
  for i = 1, grad:size(1) do
    -- TODO: do something with x[i] and evaluate f twice, and put your estimate of df/dx_i into grad_est[i]
    x[i] = x[i] + eps
    loss1 = f(x)
    x[i] = x[i] - 2 * eps
    loss2 = f(x)
    grad_est[i] = (loss1-loss2)/(2 * eps)
  end

  -- computes (symmetric) relative error of gradient
  local diff = torch.norm(grad - grad_est) / torch.norm(grad + grad_est)
  return diff, grad, grad_est
end

function fakedata(opt)
    local data = {}
    data.inputs = torch.randn(opt.batch_size, opt.seq_length, opt.input_size)                     -- random standard normal distribution for inputs
    data.targets = torch.rand(opt.batch_size,opt.seq_length):add(1):floor()  -- random integers from {1,2,3}
    return data
end

---------------------------------------------------------
-- generate fake data, then do the gradient check
--
opt={
  input_size=1,
  rnn_size=5,
  output_size=2,
  seq_length=10,
  batch_size=20,
}
torch.manualSeed(1)
local data = fakedata(opt)

local protos = create_model(opt)
-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.softmax)
params:uniform(-0.08, 0.08)

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

local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
local lstm_h = {[0]=initstate_h} -- output values of LSTM
local predictions = {}           -- softmax outputs


-- returns loss(params)
local f = function(x)
  if x ~= params then
    params:copy(x)
  end
  -- return criterion:forward(model:forward(data.inputs), data.targets)

  local loss = 0
  for t=1,opt.seq_length do
      -- we're feeding the *correct* things in here, alternatively
      -- we could sample from the previous timestep and embed that, but that's
      -- more commonly done for LSTM encoder-decoder models
      lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{data.inputs[{{},t}]
      :resize(opt.batch_size,opt.input_size), lstm_c[t-1], lstm_h[t-1]})
      predictions[t] = clones.softmax[t]:forward(lstm_h[t])
      loss = loss + clones.criterion[t]:forward(predictions[t], data.targets[{{}, t}])
  end
  return loss
end
-- returns dloss(params)/dparams
local g = function(x)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()

  -- local outputs = model:forward(data.inputs)
  -- criterion:forward(outputs, data.targets)
  -- model:backward(data.inputs, criterion:backward(outputs, data.targets))
  --

  local dx = {}                              -- d loss / d input embeddings
  local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
  local dlstm_h = {}                                  -- output values of LSTM
  for t=opt.seq_length,1,-1 do
      -- backprop through loss, and softmax/linear
      local doutput_t = clones.criterion[t]:backward(predictions[t], data.targets[{{}, t}])
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
          {data.inputs[{{},t}]
          :resize(opt.batch_size,opt.input_size), lstm_c[t-1], lstm_h[t-1]},--x,c0,h0
          {dlstm_c[t], dlstm_h[t]}
      ))

      -- backprop through embeddings
      -- clones.embed[t]:backward(x[{{}, t}], dx[t])
  end

  return grad_params
end


local diff = checkgrad(f, g, params)
print(diff)
