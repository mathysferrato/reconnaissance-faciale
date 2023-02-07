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
personne = randi(nb_personnes);
posture = randi(nb_postures);
% personne = V(randi(length(V))); % pour prendre une personne de la base d'apprentissage
% posture = randi(nb_postures_base); % pour prendre une posture dans la base d'apprentissage
% si on veut tester/mettre au point, on fixe l'individu
% personne = 10;
% posture = 6;

% Dimensions du masque
ligne_min = 200;
ligne_max = 350;
colonne_min = 60;
colonne_max = 290;

Images = zeros(1, 120000);

for j = 1:nb_personnes
	for k = 1:nb_postures
        ficF = strcat('./Data/', liste_personnes{j}, liste_postures{k}, '-300x400.gif');
        img = imread(ficF);
      
        % Degradation de l'image
        img(ligne_min:ligne_max,colonne_min:colonne_max) = 0;
        
        Images = cat(1, Images, double(transpose(img(:))));
	end
end
Images = Images(2:end,:);

if (personne == 1)
    image_test = Images(posture,:);
else
    image_test = Images((personne-1)*nb_postures + posture,:);
end



% dans un second temps, q peut être calculé afin d'atteindre le pourcentage
% d'information avec q valeurs propres (contraste)
% Pourcentage d'information 
% per = 0.75;

%% Question 5

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

for i = V
    if (i == 1)
        DataA = cat(1, DataA, Images(1:nb_postures_base,:));
    else
        DataA = cat(1, DataA, Images((i-1)*nb_postures+1:(i-1)*nb_postures + nb_postures_base,:));
    end
end
DataA = DataA(2:end,:);



% Choix du nombre de voisins
K = 1;

% Initialisation du vecteur des classes
ListeClass = 1:nb_personnes*nb_postures;

% Nombre d'images test
Nt_test = 1; % Une seule image est testée

figure('Name','Image tiree aleatoirement','Position',[0.2*L,0.2*H,0.8*L,0.5*H]);

subplot(3, 3, 2);
% Affichage de l'image de test :
colormap gray;
image_test = reshape(image_test, [400, 300]);
image_test_copy = image_test;
imagesc(image_test);
title('Image dégradée', 'FontSize', 20);
axis image;
i = 1;
% Nombre q de composantes principales à prendre en compte
Q = [2, 8, size(W,2)];
for q = Q
ind = q;
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

%Chargement de l'image proche sans masque
ficF = strcat('./Data/', liste_personnes{personne_proche}, liste_postures{posture_proche}, '-300x400.gif');
image_proche_sans_masque = imread(ficF);

image_test(ligne_min:ligne_max,colonne_min:colonne_max) = image_proche_sans_masque(ligne_min:ligne_max,colonne_min:colonne_max);

subplot(3, 3, i+3);
imagesc(image_test);
title('Image reconstruite (kppv)');
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

%Chargement de l'image proche sans masque
ficF = strcat('./Data/', liste_personnes{personne_proche}, liste_postures{posture_proche}, '-300x400.gif');
image_proche_sans_masque = imread(ficF);

image_test_copy(ligne_min:ligne_max,colonne_min:colonne_max) = image_proche_sans_masque(ligne_min:ligne_max,colonne_min:colonne_max);

subplot(3, 3, i+6);
imagesc(image_test_copy);
title('Image reconstruite (bayesien)');
axis image;
i = i+1;
end