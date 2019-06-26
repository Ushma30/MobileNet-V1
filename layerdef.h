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

#define HEIGHT_8 28
#define WIDTH_8 28
#define IP_FM_8 128
#define OP_FM_8 256
#define S1_8 0.023528477177023888
#define S2_8 0.007601846940815449
#define S3_8 0.023528477177023888
#define M_8 ((S1_1 * S2_1) / S3_1)
#define SBIAS_8 (S1_1 * S2_1) 
#define Z2_8 151

/*******************************************************************************
* Defines - Layer 9 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_9 28
#define WIDTH_9 28
#define IP_FM_9 256
#define OP_FM_9 256
#define S1_9 0.023528477177023888
#define S2_9 0.04105526953935623
#define S3_9 0.023528477177023888
#define M_9 ((S1_1 * S2_1) / S3_1)
#define SBIAS_9 (S1_1 * S2_1) 
#define Z2_9 129

/*******************************************************************************
* Defines - Layer 10 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_10 28
#define WIDTH_10 28
#define IP_FM_10 256
#define OP_FM_10 256
#define S1_10 0.023528477177023888
#define S2_10 0.006431614048779011
#define S3_10 0.023528477177023888
#define M_10 ((S1_1 * S2_1) / S3_1)
#define SBIAS_10 (S1_1 * S2_1) 
#define Z2_10 122

/*******************************************************************************
* Defines - Layer 11 Depthwise Convolution - Stride 2                          *
*******************************************************************************/

#define HEIGHT_11 28
#define WIDTH_11 28
#define IP_FM_11 256
#define OP_FM_11 256
#define S1_11 0.023528477177023888
#define S2_11 0.013460792601108551
#define S3_11 0.023528477177023888
#define M_11 ((S1_1 * S2_1) / S3_1)
#define SBIAS_11 (S1_1 * S2_1) 
#define Z2_11 122

/*******************************************************************************
* Defines - Layer 12 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_12 14
#define WIDTH_12 14
#define IP_FM_12 256
#define OP_FM_12 512
#define S1_12 0.023528477177023888
#define S2_12 0.00917122047394514
#define S3_12 0.023528477177023888
#define M_12 ((S1_1 * S2_1) / S3_1)
#define SBIAS_12 (S1_1 * S2_1) 
#define Z2_12 109

/*******************************************************************************
* Defines - Layer 13 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_13 14
#define WIDTH_13 14
#define IP_FM_13 512
#define OP_FM_13 512
#define S1_13 0.023528477177023888
#define S2_13 0.036934755742549896
#define S3_13 0.023528477177023888
#define M_13 ((S1_1 * S2_1) / S3_1)
#define SBIAS_13 (S1_1 * S2_1) 
#define Z2_13 132

/*******************************************************************************
* Defines - Layer 14 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_14 14
#define WIDTH_14 14
#define IP_FM_14 512
#define OP_FM_14 512
#define S1_14 0.023528477177023888
#define S2_14 0.005300046876072884
#define S3_14 0.023528477177023888
#define M_14 ((S1_1 * S2_1) / S3_1)
#define SBIAS_14 (S1_1 * S2_1) 
#define Z2_14 140

/*******************************************************************************
* Defines - Layer 15 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_15 14
#define WIDTH_15 14
#define IP_FM_15 512
#define OP_FM_15 512
#define S1_15 0.023528477177023888
#define S2_15 0.042609862983226776
#define S3_15 0.023528477177023888
#define M_15 ((S1_1 * S2_1) / S3_1)
#define SBIAS_15 (S1_1 * S2_1) 
#define Z2_15 94

/*******************************************************************************
* Defines - Layer 16 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_16 14
#define WIDTH_16 14
#define IP_FM_16 512
#define OP_FM_16 512
#define S1_16 0.023528477177023888
#define S2_16 0.0049632852897048
#define S3_16 0.023528477177023888
#define M_16 ((S1_1 * S2_1) / S3_1)
#define SBIAS_16 (S1_1 * S2_1) 
#define Z2_16 127

/*******************************************************************************
* Defines - Layer 17 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_17 14
#define WIDTH_17 14
#define IP_FM_17 512
#define OP_FM_17 512
#define S1_17 0.023528477177023888
#define S2_17 0.028358859941363335
#define S3_17 0.023528477177023888
#define M_17 ((S1_1 * S2_1) / S3_1)
#define SBIAS_17 (S1_1 * S2_1) 
#define Z2_17 127

/*******************************************************************************
* Defines - Layer 18 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_18 14
#define WIDTH_18 14
#define IP_FM_18 512
#define OP_FM_18 512
#define S1_18 0.023528477177023888
#define S2_18 0.007770895957946777
#define S3_18 0.023528477177023888
#define M_18 ((S1_1 * S2_1) / S3_1)
#define SBIAS_18 (S1_1 * S2_1) 
#define Z2_18 89

/*******************************************************************************
* Defines - Layer 19 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_19 14
#define WIDTH_19 14
#define IP_FM_19 512
#define OP_FM_19 512
#define S1_19 0.023528477177023888
#define S2_19 0.024329448118805885
#define S3_19 0.023528477177023888
#define M_19 ((S1_1 * S2_1) / S3_1)
#define SBIAS_19 (S1_1 * S2_1) 
#define Z2_19 134

/*******************************************************************************
* Defines - Layer 20 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_20 14
#define WIDTH_20 14
#define IP_FM_20 512
#define OP_FM_20 512
#define S1_20 0.023528477177023888
#define S2_20 0.009658650495111942
#define S3_20 0.023528477177023888
#define M_20 ((S1_1 * S2_1) / S3_1)
#define SBIAS_20 (S1_1 * S2_1) 
#define Z2_20 99

/*******************************************************************************
* Defines - Layer 21 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_21 14
#define WIDTH_21 14
#define IP_FM_21 512
#define OP_FM_21 512
#define S1_21 0.023528477177023888
#define S2_21 0.019366811960935593
#define S3_21 0.023528477177023888
#define M_21 ((S1_1 * S2_1) / S3_1)
#define SBIAS_21 (S1_1 * S2_1) 
#define Z2_21 106

/*******************************************************************************
* Defines - Layer 22 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_22 14
#define WIDTH_22 14
#define IP_FM_22 512
#define OP_FM_22 512
#define S1_22 0.023528477177023888
#define S2_22 0.005446993745863438
#define S3_22 0.023528477177023888
#define M_22 ((S1_1 * S2_1) / S3_1)
#define SBIAS_22 (S1_1 * S2_1) 
#define Z2_22 153

/*******************************************************************************
* Defines - Layer 23 Depthwise Convolution - Stride 2                          *
*******************************************************************************/

#define HEIGHT_23 14
#define WIDTH_23 14
#define IP_FM_23 512
#define OP_FM_23 512
#define S1_23 0.023528477177023888
#define S2_23 0.007835594937205315
#define S3_23 0.023528477177023888
#define M_23 ((S1_1 * S2_1) / S3_1)
#define SBIAS_23 (S1_1 * S2_1) 
#define Z2_23 126

/*******************************************************************************
* Defines - Layer 24 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_24 7
#define WIDTH_24 7
#define IP_FM_24 512
#define OP_FM_24 1024
#define S1_24 0.023528477177023888
#define S2_24 0.00817922968417406
#define S3_24 0.023528477177023888
#define M_24 ((S1_1 * S2_1) / S3_1)
#define SBIAS_24 (S1_1 * S2_1) 
#define Z2_24 130

/*******************************************************************************
* Defines - Layer 25 Depthwise Convolution - Stride 2                          *
*******************************************************************************/

#define HEIGHT_25 7
#define WIDTH_25 7
#define IP_FM_25 1024
#define OP_FM_25 1024
#define S1_25 0.023528477177023888
#define S2_25 0.12616927921772003
#define S3_25 0.023528477177023888
#define M_25 ((S1_1 * S2_1) / S3_1)
#define SBIAS_25 (S1_1 * S2_1) 
#define Z2_25 211

/*******************************************************************************
* Defines - Layer 26 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_26 7
#define WIDTH_26 7
#define IP_FM_26 1024
#define OP_FM_26 1024
#define S1_26 0.023528477177023888
#define S2_26 0.018048152327537537
#define S3_26 0.023528477177023888
#define M_26 ((S1_1 * S2_1) / S3_1)
#define SBIAS_26 (S1_1 * S2_1) 
#define Z2_26 95

/*******************************************************************************
* Defines - Layer 27 Average Pool - Stride 1                                   *
*******************************************************************************/
#define HEIGHT_27 7
#define WIDTH_27 7
#define IP_FM_27 1024
#define OP_FM_27 1024

/*******************************************************************************
* Defines - Layer 28 Fully connected Layer                                     *
*******************************************************************************/
#define ELEMENTS 1024
#define CLASSES 1000

/* 
#define FILTER_SIZE_L5 128
#define FILTER_SIZE_L9 256
#define FILTER_SIZE_L13 512
#define FILTER_SIZE_L25 1024
#define FILTER_SIZE_L29 1000
*/
#endif