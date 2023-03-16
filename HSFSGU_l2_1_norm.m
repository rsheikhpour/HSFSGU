%% HSFSGU(?2,1-norm) algorithm 
% This sript is the code for the HSFSGU(?2,1-norm) algorithm for
% semi-supervised feature selection


% Reference:
% Hessian-based Semi-supervised Feature Selection using Generalized Uncorrelated Constraint Annotation. 
% Razieh Sheikhpoura, Kamal Berahmandb, Saman Forouzandehc 
% Knowledge-Based Systems 2023.

function [Z]=HSFSGU_l2_1_norm(X_label,X_unlabel,Y_label,H,beta,lambda)
%% Input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. The training labeled data: X_label(d*n_label) with their labels Y_label(n_lable*C)
% 2. The training unlabeled data: X_unlabel(d*n_unlabel)
% 3. The Hessian matrix H(n*n) or the Hessian-Laplacian matrix HL(n*n)
% 4.Parameters
%           beta : the regularization parameter beta in HSFSGU or HLSFSGU
%           lambda : the ?2,p-norm regularization parameter lambda in HSFSGU or HLSFSGU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Output          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       The sparse projection matrix which can be used to select
%       the most relevant features.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Main script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_tr = [X_label, X_unlabel];
[d, n] = size(X_tr);
n_label = size(Y_label,1);
C = size(Y_label,2);

%% Block matrix representation of Hessian matrix H
H_ll=H(1:n_label,1:n_label);
H_lu=H(1:n_label,(n_label+1):n);  
H_ul=H((n_label+1):n,1:n_label);
H_uu=H((n_label+1):n,(n_label+1):n);

%% Compute F and Z matrices
F_label=Y_label;
F_unlabel=zeros(n-n_label,C);
F=[F_label; F_unlabel];
Z = rand(d, C);
%% Set iter and alpha, and initialize diagonal matrix D
iter = 1;
alpha = 1;
diff = 1;
D = eye(d); 
while t < 100
    %% Step 1 of Algorithm_2: Update Z by Algorithm_1 in the paper
    S_td = X_tr*X_tr' + lambda*D;
    N =((S_td)^(-1/2))*X_tr*F;
    [U,~,V] = svd(N, 0);
    M = U*V';
    Z = S_td^(-1/2)*M; 
    D = diag(1./(2*sqrt((sum(Z.^2,2)+eps))));  
    
    %%  Step 2 of Algorithm_2:Update F_unlabel
    F_unlabel = (inv(alpha^2*eye(n-n_label) + beta*H_uu)) * (alpha*X_unlabel'*Z - beta*H_ul*F_label);
    F = [F_label; F_unlabel];    

    %%  Step 2 of Algorithm_2: Compute alpha
    alpha = trace(Z'*X_tr*F)/trace(F'*F); 
    
    %% Compute the value of the objetive sunction
    obj(iter) = (norm((X_tr'*Z - alpha*F), 'fro'))^2  + beta*trace(F'*H*F) + lambda*sum(sqrt(sum(Z.*Z,2)+eps));
    if iter>1
       diff = obj_former - obj(iter);
    end
    obj_former = obj(iter);
    if diff < 10^-4
        break;
    end
    iter = iter+1;

end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(J)