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
#define S1_1 0.023528477177023888
#define S2_1 0.29219913482666016
#define S3_1 0.023528477177023888
#define M_1 ((S1_1 * S2_1) / S3_1)
#define SBIAS_1 (S1_1 * S2_1) 
#define Z2_1 110

/*******************************************************************************
* Defines - Layer 2 Pointwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_2 112
#define WIDTH_2 112
#define IP_FM_2 32
#define OP_FM_2 64
#define S1_2 0.023528477177023888
#define S2_2 0.030420949682593346
#define S3_2 0.023528477177023888
#define M_2 ((S1_1 * S2_1) / S3_1)
#define SBIAS_2 (S1_1 * S2_1) 
#define Z2_2 121

/*******************************************************************************
* Defines - Layer 3 Depthwise Convolution - Stride 2                           *
*******************************************************************************/

#define HEIGHT_3 112
#define WIDTH_3 112
#define IP_FM_3 64
#define OP_FM_3 64
#define S1_3 0.023528477177023888
#define S2_3 0.40277284383773804
#define S3_3 0.023528477177023888
#define M_3 ((S1_1 * S2_1) / S3_1)
#define SBIAS_3 (S1_1 * S2_1) 
#define Z2_3 130

/*******************************************************************************
* Defines - Layer 4 Pointwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_4 56
#define WIDTH_4 56
#define IP_FM_4 64
#define OP_FM_4 128
#define S1_4 0.023528477177023888
#define S2_4 0.015148180536925793
#define S3_4 0.023528477177023888
#define M_4 ((S1_1 * S2_1) / S3_1)
#define SBIAS_4 (S1_1 * S2_1) 
#define Z2_4 104

/*******************************************************************************
* Defines - Layer 5 Depthwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_5 56
#define WIDTH_5 56
#define IP_FM_5 128
#define OP_FM_5 128
#define S1_5 0.023528477177023888
#define S2_5 0.06053730100393295
#define S3_5 0.023528477177023888
#define M_5 ((S1_1 * S2_1) / S3_1)
#define SBIAS_5 (S1_1 * S2_1) 
#define Z2_5 160

/*******************************************************************************
* Defines - Layer 6 Pointwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_6 56
#define WIDTH_6 56
#define IP_FM_6 128
#define OP_FM_6 128
#define S1_6 0.023528477177023888
#define S2_6 0.013755458407104015
#define S3_6 0.023528477177023888
#define M_6 ((S1_1 * S2_1) / S3_1)
#define SBIAS_6 (S1_1 * S2_1) 
#define Z2_6 94

/*******************************************************************************
* Defines - Layer 7 Depthwise Convolution - Stride 2                           *
*******************************************************************************/

#define HEIGHT_7 56
#define WIDTH_7 56
#define IP_FM_7 128
#define OP_FM_7 128
#define S1_7 0.023528477177023888
#define S2_7 0.01675807684659958
#define S3_7 0.023528477177023888
#define M_7 ((S1_1 * S2_1) / S3_1)
#define SBIAS_7 (S1_1 * S2_1) 
#define Z2_7 123

/*******************************************************************************
* Defines - Layer 8 Pointwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_8 56
#define WIDTH_8 56
#define IP_FM_8 128
#define OP_FM_8 128
#define S1_8 0.023528477177023888
#define S2_8 0.01675807684659958
#define S3_8 0.023528477177023888
#define M_8 ((S1_1 * S2_1) / S3_1)
#define SBIAS_8 (S1_1 * S2_1) 
#define Z2_8 123

/*******************************************************************************
* Defines - Layer 9 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_9 56
#define WIDTH_9 56
#define IP_FM_9 128
#define OP_FM_9 128
#define S1_9 0.023528477177023888
#define S2_9 0.01675807684659958
#define S3_9 0.023528477177023888
#define M_9 ((S1_1 * S2_1) / S3_1)
#define SBIAS_9 (S1_1 * S2_1) 
#define Z2_9 123

/*******************************************************************************
* Defines - Layer 10 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_10 56
#define WIDTH_10 56
#define IP_FM_10 128
#define OP_FM_10 128
#define S1_10 0.023528477177023888
#define S2_10 0.01675807684659958
#define S3_10 0.023528477177023888
#define M_10 ((S1_1 * S2_1) / S3_1)
#define SBIAS_10 (S1_1 * S2_1) 
#define Z2_10 123

/*******************************************************************************
* Defines - Layer 11 Depthwise Convolution - Stride 2                          *
*******************************************************************************/

#define HEIGHT_11 56
#define WIDTH_11 56
#define IP_FM_11 128
#define OP_FM_11 128
#define S1_11 0.023528477177023888
#define S2_11 0.01675807684659958
#define S3_11 0.023528477177023888
#define M_11 ((S1_1 * S2_1) / S3_1)
#define SBIAS_11 (S1_1 * S2_1) 
#define Z2_11 123


/* 
#define FILTER_SIZE_L5 128
#define FILTER_SIZE_L9 256
#define FILTER_SIZE_L13 512
#define FILTER_SIZE_L25 1024
#define FILTER_SIZE_L29 1000
*/
#endif