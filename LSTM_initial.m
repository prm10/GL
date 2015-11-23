function args=LSTM_initial(args)
%% weight initial
for i1=1:length(args.layer)-2
    M=args.layer(i1);
    N=args.layer(i1+1);
    % input gates
    args.Weight{i1}.w_i=0.1/N*rand(M+1,N);
    args.Weight{i1}.r_i=0.1/N*rand(N,N);
    args.Weight{i1}.p_i=0.1/N*rand(1,N);
    % forget gates
    args.Weight{i1}.w_f=0.1/N*rand(M+1,N);
    args.Weight{i1}.r_f=0.1/N*rand(N,N);
    args.Weight{i1}.p_f=0.1/N*rand(1,N);
    % cells
    args.Weight{i1}.w_z=0.1/N*rand(M+1,N);
    args.Weight{i1}.r_z=0.1/N*rand(N,N);
    % output gates
    args.Weight{i1}.w_o=0.1/N*rand(M+1,N);
    args.Weight{i1}.r_o=0.1/N*rand(N,N);
    args.Weight{i1}.p_o=0.1/N*rand(1,N);
end
% output
args.Weight{length(args.layer)-1}.w_k=0.1/args.layer(end-1)*rand(args.layer(end-1),args.layer(end));