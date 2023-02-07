clear;
close all;

% Chargement des spectres des 4 types de matrice
load A_20_1.mat; % imat = 1
D1 = D;
load A_20_2.mat; % imat = 2
D2 = D;
load A_20_3.mat; % imat = 3
D3 = D;
load A_20_4.mat; % imat = 4
D4 = D;

% Contruction de liste d'abscisses pour les tracés
imat_abscisses = ones(n,1)*[1 2 3 4];

% Construction de la listes des spectres pour les ordonnées
sp_A = [D1 D2 D3 D4];

% Tracés
figure;

plot(imat_abscisses(:,1),sp_A(:,1), 'rx');
hold on;
plot(imat_abscisses(:,2),sp_A(:,2), 'gx');
hold on;
plot(imat_abscisses(:,3),sp_A(:,3), 'bx');
hold on;
plot(imat_abscisses(:,4),sp_A(:,4), 'kx');

xlabel("imat");
ylabel("Sp(A)");
legend ('imat = 1', 'imat = 2', 'imat = 3', 'imat = 4');

