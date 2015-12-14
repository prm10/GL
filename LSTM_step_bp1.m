function [delta_down,delta_c0,MW]=LSTM_step_bp1(args,yout,ypredict,W,MW,x,in2,f2,z2,c,o2,y,c0)
    momentum=0;%args.momentum;
    learningrate=args.learningrate;
    T=size(yout,1);
    lenO=size(yout,2);
    lenC=size(W{1}.w_i,1)-lenO-1;
    delta_up=zeros(1,lenO+lenC+1);
    for lay_i=1:length(W)-1
        delta_y{lay_i}=zeros(T,size(y{lay_i},2));
        delta_o{lay_i}=zeros(T,size(o2{lay_i},2));
        delta_c{lay_i}=zeros(T,size(c{lay_i},2));
        delta_f{lay_i}=zeros(T,size(f2{lay_i},2));
        delta_i{lay_i}=zeros(T,size(in2{lay_i},2));
        delta_z{lay_i}=zeros(T,size(z2{lay_i},2));
    end
    w_k=W{end}.w_k;
    for t=T:-1:1
        delta_k(t,:)=(-(yout(t,:)-ypredict(t,:))/size(yout,1)+delta_up(:,lenC+1:lenC+lenO)).*(1-ypredict(t,:).^2);
        delta_up=delta_k(t,:)*w_k';
        for lay_i=length(W)-1:-1:1
            if t==T
                delta_y{lay_i}(t,:)=delta_up;
                delta_o{lay_i}(t,:)=delta_y{lay_i}(t,:).*tanh(c{lay_i}(t,:)).*dsigmoid(o2{lay_i}(t,:));
                delta_c{lay_i}(t,:)=delta_y{lay_i}(t,:).*o2{lay_i}(t,:).*dtanh(tanh(c{lay_i}(t,:)))+W{lay_i}.p_o.*delta_o{lay_i}(t,:);
            else
                delta_y{lay_i}(t,:)=delta_up+delta_z{lay_i}(t+1,:)*W{lay_i}.r_z'+delta_i{lay_i}(t+1,:)*W{lay_i}.r_i'+delta_f{lay_i}(t+1,:)*W{lay_i}.r_f'+delta_o{lay_i}(t+1,:)*W{lay_i}.r_o';
                delta_o{lay_i}(t,:)=delta_y{lay_i}(t,:).*tanh(c{lay_i}(t,:)).*dsigmoid(o2{lay_i}(t,:));
                delta_c{lay_i}(t,:)=delta_y{lay_i}(t,:).*o2{lay_i}(t,:).*dtanh(tanh(c{lay_i}(t,:)))+W{lay_i}.p_o.*delta_o{lay_i}(t,:)+W{lay_i}.p_i.*delta_i{lay_i}(t+1,:)...
                    +W{lay_i}.p_f.*delta_f{lay_i}(t+1,:)+delta_c{lay_i}(t+1,:).*f2{lay_i}(t+1,:);
            end
            if t==1
                delta_f{lay_i}(t,:)=delta_c{lay_i}(t,:).*c0{lay_i}.*dsigmoid(f2{lay_i}(t,:));
                delta_c0{lay_i}=W{lay_i}.p_i.*delta_i{lay_i}(1,:)+W{lay_i}.p_f.*delta_f{lay_i}(1,:)+delta_c{lay_i}(1,:).*f2{lay_i}(1,:);
            else
                delta_f{lay_i}(t,:)=delta_c{lay_i}(t,:).*c{lay_i}(t-1,:).*dsigmoid(f2{lay_i}(t,:));
            end
            delta_i{lay_i}(t,:)=delta_c{lay_i}(t,:).*z2{lay_i}(t,:).*dsigmoid(in2{lay_i}(t,:));
            delta_z{lay_i}(t,:)=delta_c{lay_i}(t,:).*in2{lay_i}(t,:).*dtanh(z2{lay_i}(t,:));
            delta_x{lay_i}(t,:)=delta_z{lay_i}(t,:)*W{lay_i}.w_z'+delta_i{lay_i}(t,:)*W{lay_i}.w_i'+delta_f{lay_i}(t,:)*W{lay_i}.w_f'+delta_o{lay_i}(t,:)*W{lay_i}.w_o';
            delta_up=delta_x{lay_i}(t,1:end-1);
        end
    end
    
%     delta_y=restrict(delta_y);
%     delta_o=restrict(delta_o);
%     delta_c0=restrict(delta_c0);
%     delta_f=restrict(delta_f);
%     delta_i=restrict(delta_i);
%     delta_z=restrict(delta_z);
    
    dw_k=y{end}'*delta_k;
    MW{end}.w_k=momentum*MW{end}.w_k+dw_k;
    W{end}.w_k=W{end}.w_k-learningrate*MW{end}.w_k;
    db_k=sum(delta_k,1);
    MW{end}.b_k=momentum*MW{end}.b_k+db_k;
    W{end}.b_k=W{end}.b_k-learningrate*MW{end}.b_k;
    delta_down=sum(delta_x{1}(:,1:lenC));
    for lay_i=1:length(W)-1
        dw_o=x{lay_i}'*delta_o{lay_i};
        dw_f=x{lay_i}'*delta_f{lay_i};
        dw_i=x{lay_i}'*delta_i{lay_i};
        dw_z=x{lay_i}'*delta_z{lay_i};

        dr_o=y{lay_i}(1:end-1,:)'*delta_o{lay_i}(2:end,:);
        dr_f=y{lay_i}(1:end-1,:)'*delta_f{lay_i}(2:end,:);
        dr_i=y{lay_i}(1:end-1,:)'*delta_i{lay_i}(2:end,:);
        dr_z=y{lay_i}(1:end-1,:)'*delta_z{lay_i}(2:end,:);

        dp_i=sum(c{lay_i}(1:end-1,:).*delta_i{lay_i}(2:end,:))+c0{lay_i}.*delta_i{lay_i}(1,:);
        dp_f=sum(c{lay_i}(1:end-1,:).*delta_f{lay_i}(2:end,:))+c0{lay_i}.*delta_f{lay_i}(1,:);
        dp_o=sum(c{lay_i}.*delta_o{lay_i});

        MW{lay_i}.w_o=momentum*MW{lay_i}.w_o+dw_o;
        MW{lay_i}.w_f=momentum*MW{lay_i}.w_f+dw_f;
        MW{lay_i}.w_i=momentum*MW{lay_i}.w_i+dw_i;
        MW{lay_i}.w_z=momentum*MW{lay_i}.w_z+dw_z;
        MW{lay_i}.r_o=momentum*MW{lay_i}.r_o+dr_o;
        MW{lay_i}.r_f=momentum*MW{lay_i}.r_f+dr_f;
        MW{lay_i}.r_i=momentum*MW{lay_i}.r_i+dr_i;
        MW{lay_i}.r_z=momentum*MW{lay_i}.r_z+dr_z;
        MW{lay_i}.p_o=momentum*MW{lay_i}.p_o+dp_o;
        MW{lay_i}.p_f=momentum*MW{lay_i}.p_f+dp_f;
        MW{lay_i}.p_i=momentum*MW{lay_i}.p_i+dp_i;

        %% weight update
        % input gates
        W{lay_i}.w_i=W{lay_i}.w_i-learningrate*MW{lay_i}.w_i;
        W{lay_i}.r_i=W{lay_i}.r_i-learningrate*MW{lay_i}.r_i;
        W{lay_i}.p_i=W{lay_i}.p_i-learningrate*MW{lay_i}.p_i;
        % forget gates
        W{lay_i}.w_f=W{lay_i}.w_f-learningrate*MW{lay_i}.w_f;
        W{lay_i}.r_f=W{lay_i}.r_f-learningrate*MW{lay_i}.r_f;
        W{lay_i}.p_f=W{lay_i}.p_f-learningrate*MW{lay_i}.p_f;
        % cells
        W{lay_i}.w_z=W{lay_i}.w_z-learningrate*MW{lay_i}.w_z;
        W{lay_i}.r_z=W{lay_i}.r_z-learningrate*MW{lay_i}.r_z;
        % output gates
        W{lay_i}.w_o=W{lay_i}.w_o-learningrate*MW{lay_i}.w_o;
        W{lay_i}.r_o=W{lay_i}.r_o-learningrate*MW{lay_i}.r_o;
        W{lay_i}.p_o=W{lay_i}.p_o-learningrate*MW{lay_i}.p_o;
    end
function y=dsigmoid(z)
    y=z.*(1-z);
function y=dtanh(z)
    y=1-z.^2;
function delta_out=restrict(delta)
    for i1=1:length(delta)
        delta_out{i1}=max(-1*ones(size(delta{i1})),min(1*ones(size(delta{i1})),delta{i1}));
    end