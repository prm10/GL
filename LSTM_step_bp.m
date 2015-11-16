function [delta_down,args]=LSTM_step_bp(delta_up,args,lay_i,x,in2,f2,z2,c,o2,y)
    %% weight initial
    % input gates
    w_i=args.Weight{lay_i}.w_i;
    r_i=args.Weight{lay_i}.r_i;
    p_i=args.Weight{lay_i}.p_i;
    % forget gates
    w_f=args.Weight{lay_i}.w_f;
    r_f=args.Weight{lay_i}.r_f;
    p_f=args.Weight{lay_i}.p_f;
    % cells
    w_z=args.Weight{lay_i}.w_z;
    r_z=args.Weight{lay_i}.r_z;
    % output gates
    w_o=args.Weight{lay_i}.w_o;
    r_o=args.Weight{lay_i}.r_o;
    p_o=args.Weight{lay_i}.p_o;
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
    
    max_gradient=max([max(max(abs(delta_o))),max(max(abs(delta_f))),max(max(abs(delta_i))),max(max(abs(delta_z))),...
        max(max(abs(delta_y))),max(max(abs(delta_c)))])
    %% weight update
    % learning rate
    learningrate=args.learningrate;
    % input gates
    args.Weight{lay_i}.w_i=w_i-learningrate*dw_i;
    args.Weight{lay_i}.r_i=r_i-learningrate*dr_i;
    args.Weight{lay_i}.p_i=p_i-learningrate*dp_i;
    % forget gates
    args.Weight{lay_i}.w_f=w_f-learningrate*dw_f;
    args.Weight{lay_i}.r_f=r_f-learningrate*dr_f;
    args.Weight{lay_i}.p_f=p_f-learningrate*dp_f;
    % cells
    args.Weight{lay_i}.w_z=w_z-learningrate*dw_z;
    args.Weight{lay_i}.r_z=r_z-learningrate*dr_z;
    % output gates
    args.Weight{lay_i}.w_o=w_o-learningrate*dw_o;
    args.Weight{lay_i}.r_o=r_o-learningrate*dr_o;
    args.Weight{lay_i}.p_o=p_o-learningrate*dp_o;
function y=dsigmoid(z)
    y=z.*(1-z);
function y=dtanh(z)
    y=1-z.^2;