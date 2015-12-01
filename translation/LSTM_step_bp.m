function [delta_down,Weight,MomWeight]=LSTM_step_bp(args,delta_up,Weight,MomWeight,x,in2,f2,z2,c,o2,y)
momentum=args.momentum;
learningrate=args.learningrate;
    %% weight initial
    % input gates
    w_i=Weight.w_i;
    r_i=Weight.r_i;
    p_i=Weight.p_i;
    % forget gates
    w_f=Weight.w_f;
    r_f=Weight.r_f;
    p_f=Weight.p_f;
    % cells
    w_z=Weight.w_z;
    r_z=Weight.r_z;
    % output gates
    w_o=Weight.w_o;
    r_o=Weight.r_o;
    p_o=Weight.p_o;
    %%
    T=size(delta_up,1);
    t=T;
    delta_y(t,:)=delta_up(t,:);
    delta_o(t,:)=delta_y(t,:).*tanh(c(t,:)).*dsigmoid(o2(t,:));
    delta_c(t,:)=delta_y(t,:).*o2(t,:).*dtanh(tanh(c(t,:)))+p_o.*delta_o(t,:);
    delta_f(t,:)=delta_c(t,:).*c(t-1,:).*dsigmoid(f2(t,:));
    delta_i(t,:)=delta_c(t,:).*z2(t,:).*dsigmoid(in2(t,:));
    delta_z(t,:)=delta_c(t,:).*in2(t,:).*dtanh(z2(t,:));
    delta_x(t,:)=delta_z(t,:)*w_z'+delta_i(t,:)*w_i'+delta_f(t,:)*w_f'+delta_o(t,:)*w_o';
    for t=T-1:-1:1
        delta_y(t,:)=delta_up(t,:)+delta_z(t+1,:)*r_z'+delta_i(t+1,:)*r_i'+delta_f(t+1,:)*r_f'+delta_o(t+1,:)*r_o';
        delta_o(t,:)=delta_y(t,:).*tanh(c(t,:)).*dsigmoid(o2(t,:));
        delta_c(t,:)=delta_y(t,:).*o2(t,:).*dtanh(tanh(c(t,:)))+p_o.*delta_o(t,:)+p_i.*delta_i(t+1,:)...
            +p_f.*delta_f(t+1,:)+delta_c(t+1,:).*f2(t+1,:);
        if t==1
            delta_f(t,:)=0;
        else
            delta_f(t,:)=delta_c(t,:).*c(t-1,:).*dsigmoid(f2(t,:));
        end
        delta_i(t,:)=delta_c(t,:).*z2(t,:).*dsigmoid(in2(t,:));
        delta_z(t,:)=delta_c(t,:).*in2(t,:).*dtanh(z2(t,:));
        delta_x(t,:)=delta_z(t,:)*w_z'+delta_i(t,:)*w_i'+delta_f(t,:)*w_f'+delta_o(t,:)*w_o';
    end
    delta_o=restrict(delta_o);
    delta_f=restrict(delta_f);
    delta_i=restrict(delta_i);
    delta_z=restrict(delta_z);
    
    delta_down=delta_x(:,1:end-1);

    dw_o=x'*delta_o;
    dw_f=x'*delta_f;
    dw_i=x'*delta_i;
    dw_z=x'*delta_z;

    dr_o=y(1:end-1,:)'*delta_o(2:end,:);
    dr_f=y(1:end-1,:)'*delta_f(2:end,:);
    dr_i=y(1:end-1,:)'*delta_i(2:end,:);
    dr_z=y(1:end-1,:)'*delta_z(2:end,:);

    dp_i=sum(c(1:end-1,:).*delta_i(2:end,:));
    dp_f=sum(c(1:end-1,:).*delta_f(2:end,:));
    dp_o=sum(c.*delta_o);
    
    MomWeight.w_o=momentum*MomWeight.w_o+dw_o;
    MomWeight.w_f=momentum*MomWeight.w_f+dw_f;
    MomWeight.w_i=momentum*MomWeight.w_i+dw_i;
    MomWeight.w_z=momentum*MomWeight.w_z+dw_z;
    MomWeight.r_o=momentum*MomWeight.r_o+dr_o;
    MomWeight.r_f=momentum*MomWeight.r_f+dr_f;
    MomWeight.r_i=momentum*MomWeight.r_i+dr_i;
    MomWeight.r_z=momentum*MomWeight.r_z+dr_z;
    MomWeight.p_o=momentum*MomWeight.p_o+dp_o;
    MomWeight.p_f=momentum*MomWeight.p_f+dp_f;
    MomWeight.p_i=momentum*MomWeight.p_i+dp_i;
%     max_gradient=max([max(max(abs(delta_o))),max(max(abs(delta_f))),max(max(abs(delta_i))),max(max(abs(delta_z))),...
%         max(max(abs(delta_y))),max(max(abs(delta_c)))])
    %% weight update
    % input gates
    Weight.w_i=w_i-learningrate*MomWeight.w_i;
    Weight.r_i=r_i-learningrate*MomWeight.r_i;
    Weight.p_i=p_i-learningrate*MomWeight.p_i;
    % forget gates
    Weight.w_f=w_f-learningrate*MomWeight.w_f;
    Weight.r_f=r_f-learningrate*MomWeight.r_f;
    Weight.p_f=p_f-learningrate*MomWeight.p_f;
    % cells
    Weight.w_z=w_z-learningrate*MomWeight.w_z;
    Weight.r_z=r_z-learningrate*MomWeight.r_z;
    % output gates
    Weight.w_o=w_o-learningrate*MomWeight.w_o;
    Weight.r_o=r_o-learningrate*MomWeight.r_o;
    Weight.p_o=p_o-learningrate*MomWeight.p_o;
function y=dsigmoid(z)
    y=z.*(1-z);
function y=dtanh(z)
    y=1-z.^2;
function delta_out=restrict(delta)
%     delta_out=delta;
    delta_out=max(-1*ones(size(delta)),min(1*ones(size(delta)),delta));