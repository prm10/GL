require 'nn'
require 'requ'

function create_model(opt)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  -- OUR MODEL:
  --     lstm-> linear -> softmax
  local protos = {}
  -- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
  protos.lstm = LSTM.lstm(opt.input_size,opt.rnn_size)
  protos.softmax = nn.Sequential()
  :add(nn.Linear(opt.rnn_size,opt.output_size))
  :add(nn.LogSoftMax())
  protos.criterion = nn.ClassNLLCriterion()

  return protos
end

return create_model
