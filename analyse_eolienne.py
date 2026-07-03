import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict, deque


fichier = r"C:\Users\matth\cours-info\Turbine_Data_Kelmarsh_1_2017-01-01_-_2018-01-01_228.csv"

df = pd.read_csv(fichier,skiprows=9)


df_nettoye = df[["# Date and time","Wind speed (m/s)","Power (kW)"]].copy()

df_nettoye["Wind speed (m/s)"] = pd.to_numeric(df_nettoye["Wind speed (m/s)"], errors="coerce")
df_nettoye["Power (kW)"] = pd.to_numeric(df_nettoye["Power (kW)"], errors="coerce")



#suppression des lignes qui ont des zéros
df_plot = df_nettoye.dropna(subset=["Wind speed (m/s)", "Power (kW)"])


#premiers filtre physique pour supprimer les valeurs qui n'ont pas de sens
df_plot = df_plot[(df_plot["Wind speed (m/s)"] >= 0) & (df_plot["Wind speed (m/s)"] <= 60) & (df_plot["Power (kW)"] >= 0)]



plt.figure(figsize=(10, 6))

plt.scatter(df_plot["Wind speed (m/s)"], df_plot["Power (kW)"], s=3, alpha=0.3)

plt.xlabel("Vitesse du vent (m/s)")
plt.ylabel("Puissance produite (kW)")
plt.title("Puissance de l'éolienne en fonction de la vitesse du vent")
plt.grid(True)

#plt.show()



#Maintenant on va supprimer les outliers pour nettoyer la courbe.
#D'abord, on applique IQR


#On va noter comme ça les colonnes pour l'adapter facilement à d'autres bases de données dont les colonnes portent des noms différents
col_vitesse = "Wind speed (m/s)"
col_puissance = "Power (kW)"

# On copie le tableau pour ne pas abimer le tableau propre de base avec lequel on a lu le CSV
df_iqr = df_plot.copy()


#Première façon, pas fixe
# Taille des classes de vitesse
pas_vitesse = 0.2

# Création des classes de vitesse
df_iqr["classe_vitesse"] = (df_iqr[col_vitesse] // pas_vitesse) * pas_vitesse
df_iqr["classe_vitesse"] = df_iqr["classe_vitesse"].round(2)    
#sert à faire en sorte que les petites décimales ne créent pas des classes de vitesse différentes


# Calcul de Q1, Q3 et IQR pour chaque classe de vitesse et on regroupe le tout dans un tableau de paramètres
stats = df_iqr.groupby("classe_vitesse")[col_puissance].agg(q1=lambda x: x.quantile(0.25),q3=lambda x: x.quantile(0.75), nombre_points="count").reset_index()
#le reset_index sert à reprendre les bons indexes de lignes 0,1,2... et non pas les vitesses
stats["iqr"] = stats["q3"] - stats["q1"]

# Coefficient classique empirique 1.5 trouvé sur Wikipédia
coef_iqr = 1.5
coef_iqr_haut = 2

stats["borne_basse"] = stats["q1"] - coef_iqr * stats["iqr"]
stats["borne_haute"] = stats["q3"] + coef_iqr_haut * stats["iqr"]

# On ajoute ces bornes au tableau principal
df_iqr = df_iqr.merge(stats, on="classe_vitesse", how="left")


# Nombre minimal de points dans une classe pour appliquer le filtre pour éviter de l'appliquer dans les zones extrêmes ou il y a peu de point
min_points = 20

# Détection des outliers
df_iqr["outlier_iqr"] = ((df_iqr["nombre_points"] >= min_points) & ((df_iqr[col_puissance] < df_iqr["borne_basse"]) |(df_iqr[col_puissance] > df_iqr["borne_haute"])))


#on sépare le tableau en deux : le tableau des outlier (les == True), et les points valides (les == False)
df_clean = df_iqr[df_iqr["outlier_iqr"] == False].copy()
df_outliers = df_iqr[df_iqr["outlier_iqr"] == True].copy()



#Donne une idée de l'effet de ce premier nettoyage
#print("Nombre de points avant filtrage :", len(df_iqr))
#print("Nombre de points gardés :", len(df_clean))
#print("Nombre d'outliers supprimés :", len(df_outliers))
#print("Pourcentage supprimé :", round(len(df_outliers) / len(df_iqr) * 100, 2), "%")




plt.figure(figsize=(10, 6))

plt.scatter(df_iqr[col_vitesse],df_iqr[col_puissance],s=3,alpha=0.15,label="Données brutes")

plt.scatter(df_clean[col_vitesse],df_clean[col_puissance],s=3,alpha=0.4,label="Données après IQR")

plt.xlabel("Vitesse du vent (m/s)")
plt.ylabel("Puissance (kW)")
plt.title("Filtrage des outliers par IQR local")
plt.legend()
plt.grid(True)
#plt.show()









#on va encore affiner en filtrant par IQR mais cette fois pas ligne, c'est à dire avec des classes de puissance.

df_iqr_ligne = df_clean.copy()

# Taille des classes de puissance
pas_puissance = 25  # en kW

# Création des classes de puissance
df_iqr_ligne["classe_puissance"] = (df_iqr_ligne[col_puissance] // pas_puissance) * pas_puissance
df_iqr_ligne["classe_puissance"] = df_iqr_ligne["classe_puissance"].round(2)

# Calcul de Q1, Q3 et IQR pour la vitesse dans chaque classe de puissance
stats_ligne = df_iqr_ligne.groupby("classe_puissance")[col_vitesse].agg(q1=lambda x: x.quantile(0.25),q3=lambda x: x.quantile(0.75),nombre_points_ligne="count").reset_index()

stats_ligne["iqr"] = stats_ligne["q3"] - stats_ligne["q1"]



stats_ligne["borne_basse_ligne"] = stats_ligne["q1"] - coef_iqr * stats_ligne["iqr"]
stats_ligne["borne_haute_ligne"] = stats_ligne["q3"] + coef_iqr * stats_ligne["iqr"]

# On ajoute les bornes au tableau principal
df_iqr_ligne = df_iqr_ligne.merge(stats_ligne,on="classe_puissance",how="left")

min_points_ligne = 20

# Détection des outliers :
# pour une classe de puissance donnée, on regarde si la vitesse est anormale
#on définit aussi un seuil au dessus duquel on n'applique plus le filtre
seuil_puissance_iqr = 1900


df_iqr_ligne["outlier_iqr_ligne"] = ((df_iqr_ligne[col_puissance] < seuil_puissance_iqr) &(df_iqr_ligne["nombre_points_ligne"] >= min_points_ligne) &((df_iqr_ligne[col_vitesse] < df_iqr_ligne["borne_basse_ligne"]) |(df_iqr_ligne[col_vitesse] > df_iqr_ligne["borne_haute_ligne"])))

df_clean_ligne = df_iqr_ligne[df_iqr_ligne["outlier_iqr_ligne"] == False].copy()
df_outliers_ligne = df_iqr_ligne[df_iqr_ligne["outlier_iqr_ligne"] == True].copy()




plt.figure(figsize=(10, 6))

plt.scatter(df_iqr_ligne[col_vitesse],df_iqr_ligne[col_puissance],s=3,alpha=0.15,label="Données brutes")

plt.scatter(df_clean_ligne[col_vitesse],df_clean_ligne[col_puissance],s=3,alpha=0.4,label="Données après IQR par puissance (après IQR par vitesse)")

plt.xlabel("Vitesse du vent (m/s)")
plt.ylabel("Puissance (kW)")
plt.title("Filtrage des outliers par classes de puissance et vitesse")
plt.legend()
plt.grid(True)
#plt.show()










#On s'attaque maintenant au DBSCAN.
# On part des données déjà nettoyées par IQR
df_dbscan = df_clean_ligne[[col_vitesse, col_puissance]].copy()

#on met sous forme de numpy pour les calculs
X = df_dbscan[[col_vitesse, col_puissance]].to_numpy(dtype=float)


#on va normaliser afin de définir une distance qui va servir à la DBSCAN, sans pour autant changer les échelles
scale_vitesse = 21       # m/s
scale_puissance = 2000   # kW
X_norm = X / np.array([scale_vitesse, scale_puissance])




#On définit la fonction DBSCAN, faite avec l'aide de l'IA.

def dbscan(X, eps=0.05, min_samples=20):
    n = len(X)

    NON_VISITE = -99
    BRUIT = -1

    labels = np.full(n, NON_VISITE)
#On crée d'abord un tableau numpy de taille du nombre de mesure, qu'on remplit pour l'instant d'une valeur arbitraire NON_VISITE

    cell_size = eps
    grille = defaultdict(list)

#Après avoir crée une grille spatiale, on la découpe en petits carrés de taille eps*sacle_vitesse x eps*scale_puissance, remplie de valeur -99
#Cela va accélérer la recherche de voisin, plûtot que de faire des comparaisons de chaque point avec tous les autres

    for i, point in enumerate(X):
        cellule = tuple(np.floor(point / cell_size).astype(int))  #On définit des "coordonnées" pour chaque cellules
        grille[cellule].append(i)    #On remplit chaque cellule des points qu'elle contient
 
#Puis, pour chaque point on va chercher les voisins :
    def voisins(i):
        point = X[i]
        cellule = tuple(np.floor(point / cell_size).astype(int))   #On trouve d'abord la cellule du point i

        candidats = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cellule_voisine = (cellule[0] + dx, cellule[1] + dy)
                candidats.extend(grille.get(cellule_voisine, []))

        candidats = np.array(candidats)     #on remplit la liste de candidats, c'est à dire de potentiels voisin du point i

        if len(candidats) == 0:              #S'il n'y a pas de candidats, le point i est isolé, c'est un outlier
            return np.array([], dtype=int)

        distances = np.max(np.abs(X[candidats] - point), axis=1)      #on regarde ensuite la distance du point i aux candidats

        return candidats[distances <= eps]    #on garde que les vrais voisins

    cluster_id = 0

    for i in range(n):                   #On va ensuite remplir le tableau numpy de valeurs qui ont une signification :
        if labels[i] != NON_VISITE:          #Si le point j est déjà dans un cluster, on ne le retraite pas. Ca évite les boucles infinies.
            continue

        voisins_i = voisins(i)

        if len(voisins_i) < min_samples:
            labels[i] = BRUIT                       #si le point semble isolé, on lui associe la valeur arbitraire BRUIT, il est pour l'instant considéré comme outlier
        else:
            labels[i] = cluster_id                   #S'il y a assez de voisins, on lui associe un cluster, pareil si le 
            file = deque(voisins_i)

            while file:
                j = file.popleft()

                if labels[j] == BRUIT:
                    labels[j] = cluster_id

                if labels[j] != NON_VISITE:
                    continue

                labels[j] = cluster_id

                voisins_j = voisins(j)

                if len(voisins_j) >= min_samples:
                    for v in voisins_j:
                        if labels[v] in [NON_VISITE, BRUIT]:
                            file.append(v)

            cluster_id += 1

    return labels



#Puis on l'applique à notre tableau numpy, en choisissant les paramètres du DBSCAN 
#(après quelques essais, pas beaucoup car ma fonction dbscan met environ 5min à trier tous les points) :
"""eps = 0.05
min_points_dbscan = 20

labels = dbscan(X_norm,eps=eps,min_samples=min_points_dbscan)

df_dbscan["cluster_dbscan"] = labels

df_clean_dbscan = df_dbscan[df_dbscan["cluster_dbscan"] != -1].copy()
df_outliers_dbscan = df_dbscan[df_dbscan["cluster_dbscan"] == -1].copy()

print("Nombre de points avant DBSCAN :", len(df_dbscan))
print("Nombre de points gardés :", len(df_clean_dbscan))
print("Nombre d'outliers DBSCAN :", len(df_outliers_dbscan))
print("Pourcentage supprimé :", round(len(df_outliers_dbscan) / len(df_dbscan) * 100, 2), "%")




plt.figure(figsize=(10, 6))
plt.scatter(df_outliers_dbscan[col_vitesse],df_outliers_dbscan[col_puissance],s=3,alpha=0.7,label="Outliers DBSCAN")
plt.scatter(df_clean_dbscan[col_vitesse],df_clean_dbscan[col_puissance],s=3,alpha=0.35,label="Points gardés par DBSCAN")

plt.xlabel("Vitesse du vent (m/s)")
plt.ylabel("Puissance (kW)")
plt.title("Détection des outliers avec DBSCAN maison")
plt.legend()
plt.grid(True)"""
#plt.show()

#Ce trie DBSCAN est interressant à faire mais le résulatat n'est pas super. Il élimine seulement les points ésseulés des plus grandes vitesses.
#Pour ce jeu de données, le mieux est de ne faire que l'IQR en colonne et ligne, et conserver les points des plus grandes vitesses par des paramètres à la mano,
#ie ici l'application du filtre IQR par ligne uniquement jusqu'à 1900 kW.



#On va maintenant générer une courbe moyenne de la puissance en fonction de la vitesse.
"""df_final = df_clean_dbscan.copy()

pas_vitesse_courbe = 0.2

df_final["classe_vitesse_courbe"] = (df_final[col_vitesse] // pas_vitesse_courbe) * pas_vitesse_courbe

df_final["classe_vitesse_courbe"] = df_final["classe_vitesse_courbe"].round(2)

courbe_moyenne = df_final.groupby("classe_vitesse_courbe")[col_puissance].agg(puissance_moyenne="mean",puissance_mediane="median",nombre_points="count").reset_index()

min_points_courbe = 20

courbe_moyenne = courbe_moyenne[courbe_moyenne["nombre_points"] >= min_points_courbe].copy()

plt.figure(figsize=(10, 6))

plt.scatter(df_final[col_vitesse],df_final[col_puissance],s=3,alpha=0.10,label="Données nettoyées")

plt.plot(courbe_moyenne["classe_vitesse_courbe"],courbe_moyenne["puissance_moyenne"],linewidth=2,color="green",label="Courbe moyenne")

plt.plot(courbe_moyenne["classe_vitesse_courbe"],courbe_moyenne["puissance_mediane"],linewidth=2,color="orange",linestyle="--",label="Courbe médiane")

plt.xlabel("Vitesse du vent (m/s)")
plt.ylabel("Puissance (kW)")
plt.title("Courbe de puissance moyenne de l'éolienne")
plt.legend()
plt.grid(True)
plt.show()

courbe_moyenne.to_csv("courbe_moyenne_eolienne.csv", index=False)"""









#Tentative d'algo d'isolation forest
df_iso = df_clean_ligne[[col_vitesse, col_puissance]].copy()

X_iso = df_iso[[col_vitesse, col_puissance]].to_numpy(dtype=float)

X_iso_norm = X_iso / np.array([scale_vitesse, scale_puissance])

#Correction théorique utilisée dans Isolation Forest qui correspond à la profondeur moyenne attendue
#pour isoler un point dans un arbre aléatoire de n points.
def c_factor(n):
    
    if n <= 1:
        return 0
    if n == 2:
        return 1

    # Approximation du nombre harmonique
    gamma = 0.5772156649
    return 2 * (np.log(n - 1) + gamma) - 2 * (n - 1) / n




#Construire récursivement un arbre d'isolation :  
def construire_arbre(X_iso, profondeur=0, profondeur_max=8, rng=None):

    n_points, n_features = X_iso.shape

    if profondeur >= profondeur_max or n_points <= 1:
        return {"type": "feuille", "taille": n_points}

    feature = rng.integers(0, n_features)

    min_val = X_iso[:, feature].min()
    max_val = X_iso[:, feature].max()

    if min_val == max_val:
        return {"type": "feuille", "taille": n_points}

    split = rng.uniform(min_val, max_val)

    masque_gauche = X_iso[:, feature] < split

    X_gauche = X_iso[masque_gauche]
    X_droite = X_iso[~masque_gauche]

    return {"type": "noeud","feature": feature,"split": split,"gauche": construire_arbre(X_gauche, profondeur + 1, profondeur_max, rng),"droite": construire_arbre(X_droite, profondeur + 1, profondeur_max, rng)}




#Calcul de la profondeur à laquelle un point est isolé dans un arbre :
def longueur_chemin(point, arbre, profondeur=0):

    if arbre["type"] == "feuille":
        return profondeur + c_factor(arbre["taille"])

    feature = arbre["feature"]
    split = arbre["split"]

    if point[feature] < split:
        return longueur_chemin(point, arbre["gauche"], profondeur + 1)
    else:
        return longueur_chemin(point, arbre["droite"], profondeur + 1)
    


    

def isolation_forest_maison(X, n_arbres=100, taille_echantillon=256, random_state=42):
    rng = np.random.default_rng(random_state)

    n = len(X)
    profondeur_max = int(np.ceil(np.log2(taille_echantillon)))

    foret = []

    for _ in range(n_arbres):
        indices = rng.choice(n,size=min(taille_echantillon, n),replace=False)

        X_sample = X[indices]

        arbre = construire_arbre(X_sample,profondeur=0,profondeur_max=profondeur_max,rng=rng)

        foret.append(arbre)

    return foret, profondeur_max




def score_anomalie(X, foret, taille_echantillon):
    scores = []

    c_n = c_factor(taille_echantillon)

    for point in X:
        profondeurs = [longueur_chemin(point, arbre)for arbre in foret]

        profondeur_moyenne = np.mean(profondeurs)

        score = 2 ** (-profondeur_moyenne / c_n)

        scores.append(score)

    return np.array(scores)





#On applique notre algo d'isolation forest
n_arbres = 100
taille_echantillon = 256

foret, profondeur_max = isolation_forest_maison(X_iso_norm,n_arbres=n_arbres,taille_echantillon=taille_echantillon,random_state=42)

scores = score_anomalie(X_iso_norm,foret,taille_echantillon=taille_echantillon)

df_iso["score_isolation"] = scores



#on choisit un seuil d'anomalie
contamination = 0.01

n_outliers = int(contamination * len(df_iso))

df_iso["outlier_iso"] = False

indices_outliers = df_iso["score_isolation"].nlargest(n_outliers).index

df_iso.loc[indices_outliers, "outlier_iso"] = True

df_clean_iso = df_iso[df_iso["outlier_iso"] == False].copy()
df_outliers_iso = df_iso[df_iso["outlier_iso"] == True].copy()

print("Nombre de points avant Isolation Forest :", len(df_iso))
print("Nombre de points gardés :", len(df_clean_iso))
print("Nombre d'outliers Isolation Forest :", len(df_outliers_iso))
print("Pourcentage supprimé :", round(len(df_outliers_iso) / len(df_iso) * 100, 2), "%")


plt.figure(figsize=(10, 6))

plt.scatter(df_clean_iso[col_vitesse],df_clean_iso[col_puissance],s=3,alpha=0.25,label="Points gardés")

plt.scatter(df_outliers_iso[col_vitesse],df_outliers_iso[col_puissance],s=8,alpha=0.7,label="Outliers Isolation Forest")

plt.xlabel("Vitesse du vent (m/s)")
plt.ylabel("Puissance (kW)")
plt.title("Détection d'outliers par Isolation Forest")
plt.legend()
plt.grid(True)
plt.show()