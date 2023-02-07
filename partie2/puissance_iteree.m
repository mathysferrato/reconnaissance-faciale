clear variables;clc
% tolerance relative minimum pour l'ecart entre deux iteration successives 
% de la suite tendant vers la valeur propre dominante 
% (si |lambda-lambda_old|/|lambda_old|<eps, l'algo a converge)
eps = 1e-8;
% nombre d iterations max pour atteindre la convergence 
% (si i > kmax, l'algo finit)
imax = 5000; 

% Generation d une matrice rectangulaire aleatoire A de taille n x p.
% On cherche le vecteur propre et la valeur propre dominants de AA^T puis
% de A^TA
n = 15;
p = 500;
m = 5;
A = 5*randn(n,p);

% AAt, AtA sont deux matrices carrees de tailles respectives (n x n) et 
% (p x p). Elles sont appelees "equations normales" de la matrice A
AAt = A*A';
M = AAt;

V = randn(n,m);
V = mgs(V);

cv = false;
k = 0;        
acc = 0;       
H = V'*M*V;

while(~cv)
   H_old = H;
   V = M*V;
   for j = 1:m
       V(:,j) = V(:,j)/norm(V(:,j));
   end
   H = V'*M*V;
   k = k+1;
   acc = norm(M*V-V*H, 'fro')/norm(M, 'fro');
   cv = (acc < eps) || (k > imax);
end


