function str_cmd = generate_cmd(end_name,w_name)
%generate cmd that change W to w
str_cmd = strcat('w_i',end_name,'=',w_name,'.w_i;','r_i',end_name,'=',w_name,'.r_i;','p_i',end_name,'=',w_name,'.p_i;','w_f',end_name,'=',w_name,'.w_f;','r_f',end_name,'=',w_name,'.r_f;','p_f',end_name,'=',w_name,'.p_f;','w_z',end_name,'=',w_name,'.w_z;','r_z',end_name,'=',w_name,'.r_z;','w_o',end_name,'=',w_name,'.w_o;','r_o',end_name,'=',w_name,'.r_o;','p_o',end_name,'=',w_name,'.p_o;','w_k',end_name,'=',w_name,'.w_k;','b_k',end_name,'=',w_name,'.b_k;');
