#ifndef CUCORREL //Pour éviter l'inclusion recursive
#define CUCORREL
#define PARAMETERS 6 // Le nombre de paramètres (et donc de champs)
#define WIDTH 2048
#define HEIGHT 2048
#define IMG_SIZE (WIDTH*HEIGHT)
#define BLOCKSIZE 1024 //La taille du bloc pour les calculs de sommes (pour les moindres carrés)
#define LVL 6 // Le nombre d'étage de la pyramide

typedef unsigned int uint;
typedef unsigned char uchar;

void writeFields(float2*, uint, uint);
#endif //ifndef CUCORREL
