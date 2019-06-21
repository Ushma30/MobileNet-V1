#ifndef LAYERDEF
#define LAYERDEF

#define FDIM 3
#define FDIM_P 1
#define FILTER_MAX 1024

/*******************************************************************************
* Defines - Layer 0 Standard Convolution - Stride 2                            *
*******************************************************************************/

#define HEIGHT_0 224
#define WIDTH_0 224
#define IP_FM_0 3
#define OP_FM_0 32
#define S1_0 0.0078125
#define S2_0 0.02182667888700962
#define S3_0 0.023528477177023888
#define M_0 ((S1_0 * S2_0) / S3_0)
#define SBIAS_0 (S1_0 * S2_0)
#define Z1_0 128 
#define Z2_0 151

/*******************************************************************************
* Defines - Layer 1 Depthwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_1 112
#define WIDTH_1 112
#define IP_FM_1 32
#define OP_FM_1 32

/*******************************************************************************
* Defines - Layer 2 Pointwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_2 112
#define WIDTH_2 112
#define IP_FM_2 32
#define OP_FM_2 64

/*******************************************************************************
* Defines - Layer 3 Depthwise Convolution - Stride 2                           *
*******************************************************************************/

#define HEIGHT_3 112
#define WIDTH_3 112
#define IP_FM_3 64
#define OP_FM_3 64

/*******************************************************************************
* Defines - Layer 4 Pointwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_4 56
#define WIDTH_4 56
#define IP_FM_4 64
#define OP_FM_4 128

/*******************************************************************************
* Defines - Layer 5 Depthwise Convolution - Stride 1                           *
*******************************************************************************/

/* #define HEIGHT_5 56
#define WIDTH_5 56
#define FILTER_SIZE_L5 64



#define FILTER_SIZE_L5 128
#define FILTER_SIZE_L9 256
#define FILTER_SIZE_L13 512
#define FILTER_SIZE_L25 1024
#define FILTER_SIZE_L29 1000 */

#endif