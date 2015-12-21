function args=LSTM_initial(args)
%% weight initial
%encoder
args.WeightEncoder=lstm_setup(args.encoderLayer);
%decoder
args.WeightDecoder=lstm_setup(args.decoderLayer);
%predict
args.WeightPredict=lstm_setup(args.predictLayer);
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
args.Mom.WeightEncoder=lstm_mom_setup(args.encoderLayer);
%decoder
args.Mom.WeightDecoder=lstm_mom_setup(args.decoderLayer);
%predict
args.Mom.WeightPredict=lstm_mom_setup(args.predictLayer);


