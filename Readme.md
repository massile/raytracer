# Moteur de rendu (raytracing)

Un moteur de rendu photoréaliste à partir de zéro.

### Video associée:

(cliquer sur l'image)
[![](https://i.ytimg.com/vi/QYxlZ0TcbRg/sddefault.jpg)](https://www.youtube.com/watch?v=QYxlZ0TcbRg)

## Quelques nombres

 - Le calcul de l'intensité d'un pixel est massivement parallèle. De ce fait, pour des raisons de performance, Nvidia CUDA est utilisé pour lancer le programme sur le GPU.

 - L'image est subdivisée en blocs de 8x8px. Et un thread est lancé sur le GPU pour chaque pixel ! (720 000 threads au total)

 - Chaque pixel reçoit au total 400 rayons lumineux. (ce qui donne un total de 288 000 000 rayons à calculer)

 - Chaque rayon peut être réfléchi au maximum 45 fois. 

## Documentation:

### Génération d'une image:

Pour simplifier le travail, le format d'image utilisé est le PPM. L'implémentation pour écrire ce fichier
se trouve [ici](image/Ppm.h). C'est un format d'image plutôt simple à gérer (cf wikipédia)

Une classe abstraite [`Image`](image/image.h) a été créée afin de pouvoir gérer d'autres formats d'image à l'avenir.

### Objets mathématiques de base:

Comme le ray tracing repose sur l'émission de rayons lumineux, il nous faut un moyen d'en représenter un.

Un rayon peut être modélisé par:
	- un point: qui donne l'origine de l'émission du rayon
	- un vecteur: qui donne la direction de propagation du rayon

#### Vecteurs

[TODO]

[TODO]
