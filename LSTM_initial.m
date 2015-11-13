function args=LSTM_initial(args)
%% weight initial
for i1=1:length(args.layer)-2
    M=args.layer(i1);
    N=args.layer(i1+1);
    % input gates
    args.Weight{i1}.w_i=rand(M+1,N);
    args.Weight{i1}.r_i=rand(N,N);
    args.Weight{i1}.p_i=rand(1,N);
    % forget gates
    args.Weight{i1}.w_f=rand(M+1,N);
    args.Weight{i1}.r_f=rand(N,N);
    args.Weight{i1}.p_f=rand(1,N);
    % cells
    args.Weight{i1}.w_z=rand(M+1,N);
    args.Weight{i1}.r_z=rand(N,N);
    % output gates
    args.Weight{i1}.w_o=rand(M+1,N);
    args.Weight{i1}.r_o=rand(N,N);
    args.Weight{i1}.p_o=rand(1,N);
end
% output
args.Weight{length(args.layer)-1}.w_k=rand(args.layer(end-1),args.layer(end));