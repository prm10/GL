function args=LSTM_initial(args)
%% weight initial
[T,M]=size(args.input);
N=args.numblocks;
outdims=size(args.label,2);
% input gates
args.Weight.w_i=rand(M+1,N);
args.Weight.r_i=rand(N,N);
args.Weight.p_i=rand(1,N);
% forget gates
args.Weight.w_f=rand(M+1,N);
args.Weight.r_f=rand(N,N);
args.Weight.p_f=rand(1,N);
% cells
args.Weight.w_z=rand(M+1,N);
args.Weight.r_z=rand(N,N);
% output gates
args.Weight.w_o=rand(M+1,N);
args.Weight.r_o=rand(N,N);
args.Weight.p_o=rand(1,N);
% output
args.Weight.w_k=rand(N,outdims);
