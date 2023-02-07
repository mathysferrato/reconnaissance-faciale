clear;
close all;

% Constantes
nb_algos = 7;
nb_imat = 4;

% Construction du tableau de taille de matrice
taille_A = (50:50:450);

% Construction du tableau des temps
temps = zeros(nb_imat, length(taille_A), nb_algos);

% Calcul du temps d'exécution pour tous les algos

% Liste des id des algos
v_list = [10 11 12 0 1 2 3];

% tolérance
eps = 1e-8;
% nombre d'itérations max pour atteindre la convergence
maxit = 1e4;
% nombre maximum de couples propres calculés
m = 20;
percentage = .1;
% puissance à laquelle on élève v pour subspace_iter_v2 et v3
p = 5;

for i = 1:4
    imat = i;
    for j = 1:length(taille_A)
        n = taille_A(j);
        if n == 450
            m = 50;
        else
            m = 20;
        end
        for k = 1:nb_algos
            if k == 1
                genere = 1;
            else
                genere = 0;
            end
            v = v_list(k);
            [W, V, flag, t] = eigen_2022(imat, n, v, m, eps, maxit, percentage, p, genere);
            temps(i, j, k) = t;
            if temps(i,j,k) == 0
                temps(i,j,k) = temps(i,j,k) +1e-2;
            end
        end
    end
end

% Tracés
for j = 1:nb_imat
    figure('Name',"Temps de calcul pour imat = " + j);
    for i = 1:nb_algos
        semilogy(taille_A, temps(j,:,i),'-o');
        hold on;
    end
    legend('eig', 'puissance itérée', 'puissance itérée améliorée', ...
        'subspace iter v0', 'subspace iter v1', 'subspace iter v2', ...
        'subspace iter v3', 'location','northwest');
    xlabel("Taille de A");
    ylabel("Temps de calcul (s)");
    title("Temps de calcul pour imat = " + j);
end