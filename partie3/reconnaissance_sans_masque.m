clear;
close all;

load eigenfaces_part3;

% Construction vecteur numéros personnes de la base d'apprentissage
V = zeros(1,length(liste_personnes_base));
for i = 1:length(liste_personnes_base)
    V(i) = str2num(liste_personnes_base{i}(2:end));
    if (strcmp(liste_personnes_base{i}(1),"m") ~= 0)
        V(i) = V(i) + 16;
    end
end

% Tirage aléatoire d'une image de test :
personne = randi(nb_personnes)
posture = randi(nb_postures)
% personne = V(randi(length(V))); % pour prendre une personne de la base d'apprentissage
% posture = randi(nb_postures_base); % pour prendre une posture dans la base d'apprentissage
% si on veut tester/mettre au point, on fixe l'individu
%personne = 10
%posture = 6

ficF = strcat('./Data/', liste_personnes{personne}, liste_postures{posture}, '-300x400.gif')
img = imread(ficF);
image_test = double(transpose(img(:)));

% Nombre q de composantes principales à prendre en compte 
q = 2;

% dans un second temps, q peut être calculé afin d'atteindre le pourcentage
% d'information avec q valeurs propres (contraste)
% Pourcentage d'information 
% per = 0.75;

%% Question 3

% Construction de DataT, DataA et LabelA
DataT = image_test;
DataA = zeros(1, length(image_test));
LabelA = zeros(1,nb_personnes_base*nb_postures_base);
k = 1;
for p = V
    for j = 1:nb_postures_base
        LabelA(k) = (p-1)*nb_postures + j;
        k = k+1;
    end
end

for i = 1:nb_personnes_base
    for j = 1:nb_postures_base
        ficF = strcat('./Data/', liste_personnes_base{i}, liste_postures{j}, '-300x400.gif');
        img_DataA = imread(ficF);
        DataA = cat(1, DataA, double(transpose(img_DataA(:))));
    end
end
DataA = DataA(2:end,:);

% Choix du nombre de voisins
K = 1;

% Initialisation du vecteur des classes
ListeClass = 1:nb_personnes*nb_postures;

% Nombre d'images test
Nt_test = 1; % Une seule image est testée

% Classement par l'algorithme des k-ppv
[Partition] = kppv(DataA, LabelA, DataT, Nt_test, K, ListeClass, W_masque, q);

% individu pseudo-résultat pour l'affichage (A CHANGER)
personne_proche = floor(Partition/nb_postures);
posture_proche = mod(Partition,nb_postures);

if (posture_proche == 0)
    posture_proche = nb_postures_base;
end

if (posture_proche < nb_postures)
    personne_proche = personne_proche +1;
end

figure('Name','Image tiree aleatoirement','Position',[0.2*L,0.2*H,0.8*L,0.5*H]);

subplot(1, 2, 1);
% Affichage de l'image de test :
colormap gray;
imagesc(img);
title({['Individu de test : posture ' num2str(posture) ' de ', liste_personnes{personne}]}, 'FontSize', 20);
axis image;

ficF = strcat('./Data/', liste_personnes{personne_proche}, liste_postures{posture_proche}, '-300x400.gif')
imgR = imread(ficF);
        
subplot(1, 2, 2);
imagesc(imgR);
title({['Individu le plus proche (kppv) : posture ' num2str(posture_proche) ' de ', liste_personnes{personne_proche}]}, 'FontSize', 20);
axis image;

%% BAYESIEN

% Classement par la classification bayesienne
[Partition] = bayesien(DataA, LabelA, DataT, Nt_test, W_masque, q);

% individu pseudo-résultat pour l'affichage (A CHANGER)
personne_proche = floor(Partition/nb_postures);
posture_proche = mod(Partition,nb_postures);

if (posture_proche == 0)
    posture_proche = nb_postures_base;
end

if (posture_proche < nb_postures)
    personne_proche = personne_proche +1;
end

figure('Name','Image tiree aleatoirement','Position',[0.2*L,0.2*H,0.8*L,0.5*H]);

subplot(1, 2, 1);
% Affichage de l'image de test :
colormap gray;
imagesc(img);
title({['Individu de test : posture ' num2str(posture) ' de ', liste_personnes{personne}]}, 'FontSize', 20);
axis image;

ficF = strcat('./Data/', liste_personnes{personne_proche}, liste_postures{posture_proche}, '-300x400.gif')
imgR = imread(ficF);
        
subplot(1, 2, 2);
imagesc(imgR);
title({['Individu la plus proche (bayesien) : posture ' num2str(posture_proche) ' de ', liste_personnes{personne_proche}]}, 'FontSize', 20);
axis image;

%% Matrice de confusion
Nt_test = 100;
grp_connus = zeros(Nt_test,1);
mat_taux_erreur_kppv = zeros(1,size(W,2)-1);
mat_taux_erreur_bayesien = zeros(1,size(W,2)-1);

% Consitution des groupes connus et de DataT
for i = 1:Nt_test
    personne = randi(nb_personnes);
    posture = randi(nb_postures);
%     personne = V(randi(length(V))); % pour prendre une personne de la base d'apprentissage
%     posture = randi(nb_postures_base); % pour prendre une posture dans la base d'apprentissage
    ficF = strcat('./Data/', liste_personnes{personne}, liste_postures{posture}, '-300x400.gif');
    img = imread(ficF);
    image_test = double(transpose(img(:)));
    DataT = cat(1, DataT, image_test);
    grp_connus(i) = (personne-1)*nb_postures + posture;
end
DataT = DataT(2:end,:);

for q = 2:size(W,2)

        % Consitution des groupes prédits
        [Partition_kppv] = kppv(DataA, LabelA, DataT, Nt_test, K, ListeClass, W_masque, q);
        grp_predits_kppv = Partition_kppv;
        [Partition_bayesien] = bayesien(DataA, LabelA, DataT, Nt_test, W_masque, q);
        grp_predits_bayesien = Partition_bayesien;
        
        mat_confusion_kppv = confusionmat(grp_connus, grp_predits_kppv);
        mat_confusion_bayesien = confusionmat(grp_connus, grp_predits_bayesien);
        taux_erreur_kppv = 0;
        taux_erreur_bayesien = 0;

        [n,p] = size(mat_confusion_kppv);
        for i = 1:n
            for j = 1:p
                if (j~= i && mat_confusion_kppv(i,j) ~= 0)
                    taux_erreur_kppv = taux_erreur_kppv + 1;
                end
            end
        end

        [n,p] = size(mat_confusion_bayesien);
        for i = 1:n
            for j = 1:p
                if (j~= i && mat_confusion_bayesien(i,j) ~= 0)
                    taux_erreur_bayesien = taux_erreur_bayesien + 1;
                end
            end
        end
        
        % Taux d'erreur pour l'exloitation des résultats
        mat_taux_erreur_kppv(q-1) = taux_erreur_kppv;
        mat_taux_erreur_bayesien(q-1) = taux_erreur_bayesien;
end

figure;
plot((2:size(W,2)), mat_taux_erreur_kppv);
hold on;
plot((2:size(W,2)), mat_taux_erreur_bayesien);
xlabel("q");
ylabel("% d'erreur dans la matrice de confusion");
legend('kppv', 'bayesien');
axis([2 size(W,2) 0 100]);

