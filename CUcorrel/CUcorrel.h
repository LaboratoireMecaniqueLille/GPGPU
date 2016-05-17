#ifndef CUCORREL //Pour éviter l'inclusion recursive
#define CUCORREL
#define WIDTH 2048
#define HEIGHT 2048
#define NTILES 16
#define IMG_SIZE (WIDTH*HEIGHT)
#define T_WIDTH (WIDTH/NTILES)
#define T_HEIGHT (HEIGHT/NTILES)
#define T_SIZE (T_WIDTH*T_HEIGHT)
#define BLOCKSIZE 1024 //La taille du bloc pour les calculs de sommes (pour les moindres carrés)
#define LVL 4 // Le nombre d'étage de la pyramide
#define DEBUG(x) cout << x << endl
//#define DEBUG(x)


typedef unsigned int uint;
typedef unsigned char uchar;

void writeFields(float2*, uint, uint);
#endif //ifndef CUCORREL
