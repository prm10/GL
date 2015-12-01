function args=LSTM_initial(args)
%% weight initial
%encoder
for i1=1:length(args.encoderLayer)-2
    M=args.encoderLayer(i1);
    N=args.encoderLayer(i1+1);
    % input gates
    args.WeightEncoder{i1}.w_i=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightEncoder{i1}.r_i=1/N*normrnd(0,0.1,[N,N]);
    args.WeightEncoder{i1}.p_i=zeros(1,N);
    % forget gates
    args.WeightEncoder{i1}.w_f=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightEncoder{i1}.r_f=1/N*normrnd(0,0.1,[N,N]);
    args.WeightEncoder{i1}.p_f=zeros(1,N);
    % cells
    args.WeightEncoder{i1}.w_z=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightEncoder{i1}.r_z=1/N*normrnd(0,0.1,[N,N]);
    % output gates
    args.WeightEncoder{i1}.w_o=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightEncoder{i1}.r_o=1/N*normrnd(0,0.1,[N,N]);
    args.WeightEncoder{i1}.p_o=zeros(1,N);
end
% translation
M=args.encoderLayer(end-1);
N=args.encoderLayer(end);
args.WeightEncoder{length(args.encoderLayer)-1}.w_k=1/M*normrnd(0,0.1,[M,N]);
args.WeightEncoder{length(args.encoderLayer)-1}.b_k=zeros(1,N);

%decoder
for i1=1:length(args.decoderLayer)-2
    M=args.decoderLayer(i1);
    N=args.decoderLayer(i1+1);
    % input gates
    args.WeightDecoder{i1}.w_i=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightDecoder{i1}.r_i=1/N*normrnd(0,0.1,[N,N]);
    args.WeightDecoder{i1}.p_i=zeros(1,N);
    % forget gates
    args.WeightDecoder{i1}.w_f=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightDecoder{i1}.r_f=1/N*normrnd(0,0.1,[N,N]);
    args.WeightDecoder{i1}.p_f=zeros(1,N);
    % cells
    args.WeightDecoder{i1}.w_z=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightDecoder{i1}.r_z=1/N*normrnd(0,0.1,[N,N]);
    % output gates
    args.WeightDecoder{i1}.w_o=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightDecoder{i1}.r_o=1/N*normrnd(0,0.1,[N,N]);
    args.WeightDecoder{i1}.p_o=zeros(1,N);
end
% output
M=args.decoderLayer(end-1);
N=args.decoderLayer(end);
args.WeightDecoder{length(args.decoderLayer)-1}.w_k=1/M*normrnd(0,0.1,[M,N]);
args.WeightDecoder{length(args.decoderLayer)-1}.b_k=zeros(1,N);

%predict
for i1=1:length(args.predictLayer)-2
    M=args.predictLayer(i1);
    N=args.predictLayer(i1+1);
    % input gates
    args.WeightPredict{i1}.w_i=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightPredict{i1}.r_i=1/N*normrnd(0,0.1,[N,N]);
    args.WeightPredict{i1}.p_i=zeros(1,N);
    % forget gates
    args.WeightPredict{i1}.w_f=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightPredict{i1}.r_f=1/N*normrnd(0,0.1,[N,N]);
    args.WeightPredict{i1}.p_f=zeros(1,N);
    % cells
    args.WeightPredict{i1}.w_z=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightPredict{i1}.r_z=1/N*normrnd(0,0.1,[N,N]);
    % output gates
    args.WeightPredict{i1}.w_o=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    args.WeightPredict{i1}.r_o=1/N*normrnd(0,0.1,[N,N]);
    args.WeightPredict{i1}.p_o=zeros(1,N);
end
% output
M=args.predictLayer(end-1);
N=args.predictLayer(end);
args.WeightPredict{length(args.predictLayer)-1}.w_k=1/M*normrnd(0,0.1,[M,N]);
args.WeightPredict{length(args.predictLayer)-1}.b_k=zeros(1,N);

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
for i1=1:length(args.encoderLayer)-2
    M=args.encoderLayer(i1);
    N=args.encoderLayer(i1+1);
    % input gates
    args.Mom.WeightEncoder{i1}.w_i=zeros(M+1,N);
    args.Mom.WeightEncoder{i1}.r_i=zeros(N,N);
    args.Mom.WeightEncoder{i1}.p_i=zeros(1,N);
    % forget gates
    args.Mom.WeightEncoder{i1}.w_f=zeros(M+1,N);
    args.Mom.WeightEncoder{i1}.r_f=zeros(N,N);
    args.Mom.WeightEncoder{i1}.p_f=zeros(1,N);
    % cells
    args.Mom.WeightEncoder{i1}.w_z=zeros(M+1,N);
    args.Mom.WeightEncoder{i1}.r_z=zeros(N,N);
    % output gates
    args.Mom.WeightEncoder{i1}.w_o=zeros(M+1,N);
    args.Mom.WeightEncoder{i1}.r_o=zeros(N,N);
    args.Mom.WeightEncoder{i1}.p_o=zeros(1,N);
end
% translation
M=args.encoderLayer(end-1);
N=args.encoderLayer(end);
args.Mom.WeightEncoder{length(args.encoderLayer)-1}.w_k=zeros(M,N);
args.Mom.WeightEncoder{length(args.encoderLayer)-1}.b_k=zeros(1,N);

%decoder
for i1=1:length(args.decoderLayer)-2
    M=args.decoderLayer(i1);
    N=args.decoderLayer(i1+1);
    % input gates
    args.Mom.WeightDecoder{i1}.w_i=zeros(M+1,N);
    args.Mom.WeightDecoder{i1}.r_i=zeros(N,N);
    args.Mom.WeightDecoder{i1}.p_i=zeros(1,N);
    % forget gates
    args.Mom.WeightDecoder{i1}.w_f=zeros(M+1,N);
    args.Mom.WeightDecoder{i1}.r_f=zeros(N,N);
    args.Mom.WeightDecoder{i1}.p_f=zeros(1,N);
    % cells
    args.Mom.WeightDecoder{i1}.w_z=zeros(M+1,N);
    args.Mom.WeightDecoder{i1}.r_z=zeros(N,N);
    % output gates
    args.Mom.WeightDecoder{i1}.w_o=zeros(M+1,N);
    args.Mom.WeightDecoder{i1}.r_o=zeros(N,N);
    args.Mom.WeightDecoder{i1}.p_o=zeros(1,N);
end
% output
M=args.decoderLayer(end-1);
N=args.decoderLayer(end);
args.Mom.WeightDecoder{length(args.decoderLayer)-1}.w_k=zeros(M,N);
args.Mom.WeightDecoder{length(args.decoderLayer)-1}.b_k=zeros(1,N);

%predict
for i1=1:length(args.predictLayer)-2
    M=args.predictLayer(i1);
    N=args.predictLayer(i1+1);
    % input gates
    args.Mom.WeightPredict{i1}.w_i=zeros(M+1,N);
    args.Mom.WeightPredict{i1}.r_i=zeros(N,N);
    args.Mom.WeightPredict{i1}.p_i=zeros(1,N);
    % forget gates
    args.Mom.WeightPredict{i1}.w_f=zeros(M+1,N);
    args.Mom.WeightPredict{i1}.r_f=zeros(N,N);
    args.Mom.WeightPredict{i1}.p_f=zeros(1,N);
    % cells
    args.Mom.WeightPredict{i1}.w_z=zeros(M+1,N);
    args.Mom.WeightPredict{i1}.r_z=zeros(N,N);
    % output gates
    args.Mom.WeightPredict{i1}.w_o=zeros(M+1,N);
    args.Mom.WeightPredict{i1}.r_o=zeros(N,N);
    args.Mom.WeightPredict{i1}.p_o=zeros(1,N);
end
% output
M=args.predictLayer(end-1);
N=args.predictLayer(end);
args.Mom.WeightPredict{length(args.predictLayer)-1}.w_k=zeros(M,N);
args.Mom.WeightPredict{length(args.predictLayer)-1}.b_k=zeros(1,N);
