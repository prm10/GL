function args=LSTM_initial(args)
%% weight initial
%encoder
args.WeightEncoder=lstm_initial(args.encoderLayer);
%decoder
args.WeightDecoder=lstm_initial(args.decoderLayer);
%predict
args.WeightPredict=lstm_initial(args.predictLayer);
% transition
for i1=1:length(args.decoderLayer)-2
    M=args.encoderLayer(end);
    N=args.decoderLayer(i1+1);
    args.WeightTranR{i1}.w_k=1/M*normrnd(0,0.1,[M,N]);
    args.WeightTranR{i1}.b_k=zeros(1,N);
    args.Mom.WeightTranR{i1}.w_k=zeros(M,N);
    args.Mom.WeightTranR{i1}.b_k=zeros(1,N);
end
for i1=1:length(args.predictLayer)-2
    M=args.encoderLayer(end);
    N=args.predictLayer(i1+1);
    args.WeightTranP{i1}.w_k=1/M*normrnd(0,0.1,[M,N]);
    args.WeightTranP{i1}.b_k=zeros(1,N);
    args.Mom.WeightTranP{i1}.w_k=zeros(M,N);
    args.Mom.WeightTranP{i1}.b_k=zeros(1,N);
end
%% Momentum initial
%encoder
args.Mom.WeightEncoder=lstm_mom_initial(args.encoderLayer);
%decoder
args.Mom.WeightDecoder=lstm_mom_initial(args.decoderLayer);
%predict
args.Mom.WeightPredict=lstm_mom_initial(args.predictLayer);

function args=lstm_initial(layers)
n=length(layers)-1;%lstm+tanh的层数
args=cell(n,1);
for i1=1:n-1
    M=layers(i1);
    N=layers(i1+1);
    % input gates
    args{i1}.w_i=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args{i1}.r_i=1/N*normrnd(0,0.1,[N,N]);
    args{i1}.p_i=zeros(1,N);
    % forget gates
    args{i1}.w_f=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args{i1}.r_f=1/N*normrnd(0,0.1,[N,N]);
    args{i1}.p_f=zeros(1,N);
    % cells
    args{i1}.w_z=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args{i1}.r_z=1/N*normrnd(0,0.1,[N,N]);
    % output gates
    args{i1}.w_o=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args{i1}.r_o=1/N*normrnd(0,0.1,[N,N]);
    args{i1}.p_o=zeros(1,N);
end
M=layers(n);
N=layers(n+1);
args{n}.w_k=1/M*normrnd(0,0.1,[M,N]);
args{n}.b_k=zeros(1,N);

function args=lstm_mom_initial(layers)
%% Momentum initial
n=length(layers)-1;%lstm+tanh的层数
args=cell(n,1);
for i1=1:n-1
    M=layers(i1);
    N=layers(i1+1);
    % input gates
    args{i1}.w_i=zeros(M+1,N);
    args{i1}.r_i=zeros(N,N);
    args{i1}.p_i=zeros(1,N);
    % forget gates
    args{i1}.w_f=zeros(M+1,N);
    args{i1}.r_f=zeros(N,N);
    args{i1}.p_f=zeros(1,N);
    % cells
    args{i1}.w_z=zeros(M+1,N);
    args{i1}.r_z=zeros(N,N);
    % output gates
    args{i1}.w_o=zeros(M+1,N);
    args{i1}.r_o=zeros(N,N);
    args{i1}.p_o=zeros(1,N);
end
% translation
M=layers(n);
N=layers(n+1);
args{n}.w_k=zeros(M,N);
args{n}.b_k=zeros(1,N);
