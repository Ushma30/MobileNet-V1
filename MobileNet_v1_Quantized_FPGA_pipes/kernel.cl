pipe unsigned char pipePath0 __attribute__((xcl_reqd_pipe_depth(32768)));
pipe unsigned char pipePath1 __attribute__((xcl_reqd_pipe_depth(32768)));

//kernel __attribute__ ((reqd_work_group_size(16, 16, 1)))
__kernel void convolute(__global unsigned char* output, 
						__global unsigned char* inp_image_r, 
						__global unsigned char* inp_image_g, 
						__global unsigned char* inp_image_b, 
						__global unsigned char* filter_k,
						__global int* bias,
						int rows, int cols, int filtersize, int stride, int op_size,
						int Q, float Sbias, unsigned char Z1, unsigned char Z2, int right_shift ) {

	int tx = get_global_id(0);
	int ty = get_global_id(1);
	int half_filtersize = (filtersize)/2;
    unsigned char pipeIp;
	int sum = 0;
	int xindex=0, yindex=0, findex=0, filter_count=0;
	int i,j,l;

	while (filter_count < op_size) {
	
		int output_shift = (rows / 2) * (cols / 2) * filter_count;
		
		for(i = 0; i < filtersize; i++){
			yindex = ty * stride + i;
			for(j = 0; j < filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex >= cols || xindex >= rows) {
					sum +=  0 * filter_k[findex];
				}
				else {
					// if ((tx == 0 && ty == 0) && filter_count == 2) {
					// 	printf("Image Index r - y: %d\t x: %d\n",yindex, xindex);
					// 	printf("Image r: %d\t filter index %d \t%d\n",inp_image_r[yindex * get_global_size(0) * stride + xindex], findex, filter_k[findex]);
					// 	printf("Image stracted r: %d\t filter index %d \t%d\n",(inp_image_r[yindex * get_global_size(0) * stride + xindex] - Z1), findex, (filter_k[findex]-Z2));
					// 	printf("Multiplication R %d\n",(inp_image_r[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2));
					// }
 						sum +=  (inp_image_r[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2);
				}
			}
		}
		for(i = 0; i < filtersize; i++){
			yindex = ty * stride + i;
			for(j = 0; j < filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex >= cols || xindex >= rows) {
					sum +=  0 * filter_k[findex];
				}
				else {
					// if ((tx == 0 && ty == 0) && filter_count == 2) {
					// 	printf("Image Index g - y: %d\t x: %d\n",yindex, xindex);
					// 	printf("Image g: %d\t filter index %d \t%d\n",inp_image_g[yindex * get_global_size(0) * stride + xindex], findex, filter_k[findex]);
					// 	printf("Image substracted g: %d\t filter index %d \t%d\n",(inp_image_g[yindex * get_global_size(0) * stride + xindex] - Z1), findex, (filter_k[findex]-Z2));
					// 	printf("Multiplication G %d\n",(inp_image_g[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2));
					// }
 					sum +=  (inp_image_g[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2);
				}
			}
		}
		for(i = 0; i < filtersize; i++){
			yindex = ty * stride + i;
			for(j = 0; j < filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex >= cols || xindex >= rows) {
					sum +=  0 * filter_k[findex];
				}
				else {
					// if ((tx == 0 && ty == 0) && filter_count == 2) {
					// 	printf("Image Index b - y: %d\t x: %d\n",yindex, xindex);
					// 	printf("Image b: %d\t filter index %d \t%d\n",inp_image_b[yindex * get_global_size(0) * stride + xindex], findex, filter_k[findex]);
					// 	printf("Image substracted b: %d\t filter index %d \t%d\n",(inp_image_b[yindex * get_global_size(0) * stride + xindex] - Z1), findex, (filter_k[findex]-Z2));
					// 	printf("Multiplication B %d\n",(inp_image_b[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2));
					// }
 					sum +=  (inp_image_b[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2);
				}
			}
		}
		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		// 	printf("B Sum: %d\n",(sum));
		// 	printf("Bias %d\n", bias[filter_count]);
		// }
		sum = sum + bias[filter_count];
		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		// 	printf("Bias added sum: %d\n",(sum));
		// }
		sum = sum * (float)Q / (float)2147483648;
		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		// 	printf("Q: %d\n",Q);
		// 	printf("Q/2^31: %f\n", (float)Q / (float)2147483648);
		// 	printf("sum * Q/2^31: %d\n",(sum));
		// }
		sum = sum + ((right_shift < 1) ? 0 : (1 << (right_shift - 1)));
		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		// 	printf("Sum + rounding: %d\n",(sum));
		// }
		sum = sum >> right_shift;
		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		// 	printf("Sum rightshifted: %d\n",(sum));
		// }
		
		if (sum <= 0) {
			sum = 0;		
		} else if (sum >= 255) 
			sum = 255;

//		if ((tx == 17 && ty == 0) ) {
//		 	printf("A Sum: %d\n",(sum));
//		}
		
		//output[(ty * get_global_size(0) + tx) + output_shift] = (unsigned char)sum;
        pipeIp = (unsigned char)sum;
        write_pipe_block(pipePath0, &pipeIp);
		sum = 0;
		filter_count++;
	}
}
//kernel __attribute__ ((reqd_work_group_size(7, 7, 1)))
__kernel void depthwise(__global unsigned char* output, 
						__global unsigned char* inp_image, 
						__global unsigned char* filter_k, 
						__global int* bias, 
						int rows, int cols, int filtersize, int stride, int op_size,
						int Q, float Sbias, unsigned char Z2, int right_shift) { 

	int tx = get_global_id(0);
	int ty = get_global_id(1);
    unsigned char pipeIp;
	int half_filtersize = (filtersize)/2;

	int sum = 0;
	int xindex=0, yindex=0, findex=0, filter_count=0, start, end;
	int i,j,l;
    int l_i, l_j;
    int localWorkSize = 16;
    __local unsigned char pipeinput[802816];
//    if (get_global_size(0) == 56 )
//    {
//        if (tx == 0 && ty == 0)
//        {
//        
//            printf("Iam in 2depthwise layer\n");
//            printf("op_size %d\n",op_size);
//            printf("rows %d\n",rows);
//            printf("cols %d\n",cols);
//        }
//    }
    if (tx == 0 && ty == 0)
    {
        for ( i = 0; i < (rows*cols); i++ )
        {
            for ( j = 0; j < op_size; j++ )
            {
                read_pipe_block(pipePath0, &pipeinput[(j*rows*cols)+i]);		    
            }
        }
//        if (get_global_size(0) == 56 )
//        {        
//            for ( i = 0; i < (112*112); i++ )
//            {
//                printf("%d\t",pipeinput[i]);	
//            }
//        }
//        for ( i = (112*112); i < (112*112*2); i++ )
//        {
//            printf("%d\t",pipeinput[i]);	
//        }
    }    
    
//    int pi;
//    if (tx == 0 && ty == 0)
//    {
//        for ( i = 0; i < 7; i++ )
//        {
//            #pragma HLS unroll factor=2
//            for ( pi = 0; pi < 7; pi++ )
//            {
//                #pragma HLS unroll factor=2
//                for (l_i = 0; l_i < localWorkSize; l_i++)
//                {
//		            #pragma HLS unroll factor=2
//                    for (l_j = 0; l_j < localWorkSize; l_j++)
//                    {
// 		                #pragma HLS unroll factor=2 
//                        for ( j = 0; j < op_size; j++ )
//                        {
//			                #pragma HLS unroll factor=2
//                            read_pipe_block(pipePath0, &pipeinput[(j*rows*cols)+( l_i*cols + l_j )+(pi*localWorkSize)+(i*localWorkSize*cols)]);		    
//                        }
//                    }
//                }
//            }
//        }   
//        /*for ( i = (112*112); i < (112*112*2); i++ )
//        {
//            printf("%d\t",pipeinput[i]);	
//        }*/
//    }
    //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    //pipeinput[(112*112)+1] = 128;
    //printf("Read Pipe Done \n");
    //printf("value %d\n",pipeinput[112*112]);
	if (stride == 1) {
		start = -half_filtersize;
		end = half_filtersize;
	} else if (stride == 2) {
		start = 0;
		end = filtersize - 1;
	}
	while (filter_count < op_size) {
		int output_shift = (rows / stride) * (cols / stride) * filter_count;
		
		for(i = start; i <= end; i++){
			yindex = ty * stride + i;
			for(j = start; j <= end; j++,findex++){
				xindex = tx * stride + j;

				if (stride == 1) {
					if ((yindex < 0 || xindex < 0) || (yindex >= cols || xindex >= rows)) {
						sum +=  0 * filter_k[findex];
					}
					else {
						// if ((tx == 0 && ty == 0) && filter_count == 0) {
						// 	printf("Image data: %d\t filter %d\n",inp_image[yindex * get_global_size(0) * stride + xindex], filter_k[findex]);
						// 	printf("Img data: %d\tfilter index %d\t%d\n",inp_image[(yindex * get_global_size(0) * stride + xindex) + (rows * cols * filter_count)], findex, (filter_k[findex] - Z2));
						// 	printf("Multiplication: %d\n",inp_image[yindex * get_global_size(0) * stride + xindex] * filter_k[findex]);
						// }
						sum +=  pipeinput[(yindex * get_global_size(0) * stride + xindex) + (rows * cols * filter_count)] * (filter_k[findex] - Z2);
					}
				} else if (stride == 2) {
					if (yindex >= cols || xindex >= rows) {
						sum +=  0 * filter_k[findex];
					}
					else {
						// if ((tx == 0 && ty == 0) && filter_count == 0) {
						// 	printf("Image data: %d\t filter %d\n",inp_image[yindex * get_global_size(0) * stride + xindex], filter_k[findex]);
						// 	printf("Img data: %d\tfilter index %d\t%d\n",inp_image[(yindex * get_global_size(0) * stride + xindex) + (rows * cols * filter_count)], findex, (filter_k[findex] - Z2));
						// 	printf("Multiplication: %d\n",inp_image[(yindex * get_global_size(0) * stride + xindex) + (rows * cols * filter_count)] * (filter_k[findex] - Z2));
						// }
						sum +=  pipeinput[(yindex * get_global_size(0) * stride + xindex) + (rows * cols * filter_count)] * (filter_k[findex] - Z2);
					}
				}
			}
		}
		// if ((tx == 0 && ty == 0) && filter_count == 0) {
		// 	printf("B Sum: %d\n",(sum));
		// }
		sum = sum + bias[filter_count];
		// if ((tx == 0 && ty == 0) && filter_count == 0) {
		// 	printf("Bias added sum: %d\n",(sum));
		// }
		sum = sum * ((float)Q / (float)2147483648);
		// if ((tx == 0 && ty == 0) && filter_count == 0) {
		// 	printf("Q: %d\n",Q);
		// 	printf("Q/2^31: %f\n", (float)Q / (float)2147483648);
		// 	printf("sum * Q/2^31: %d\n",(sum));
		// 	printf("Right shift: %d\n", (right_shift < 1) ? 0 : (1 << (right_shift - 1)));
		// }
		sum = sum + ((right_shift < 1) ? 0 : (1 << (right_shift - 1)));
		// if ((tx == 0 && ty == 0) && filter_count == 0) {
		// 	printf("Sum + rounding: %d\n",(sum));
		// }
		sum = sum >> right_shift;
		// if ((tx == 0 && ty == 0) && filter_count == 0) {
		// 	printf("Sum rightshifted: %d\n",(sum));
		// }

		if (sum <= 0) {
			sum = 0;		
		} else if (sum >= 255) 
			sum = 255;
    
//        if (get_global_size(0) == 56 )
//        {
//		    output[(ty * get_global_size(0) + tx) + output_shift] = (unsigned char)sum;
//        }
//        else
//        {
            pipeIp = (unsigned char)sum;
            write_pipe_block(pipePath1, &pipeIp);
//        }        
        sum = 0;
		filter_count++;	
	}
}

//kernel __attribute__ ((reqd_work_group_size(7, 7, 1)))
__kernel void pointwise(__global unsigned char* output, 
						__global unsigned char* inp_image, 
						__global unsigned char* filter_k, 
						__global int* bias, 
						int rows, int cols, int filtersize, int op_size,
						int Q, float Sbias, unsigned char Z2, int right_shift ) {  

	int tx = get_global_id(0);
	int ty = get_global_id(1);
    unsigned char pipeIp;
	int sum = 0;
	int findex=0, filter_count=0;
	int i,j,l;
    int localWorkSize = 7;
    int l_i, l_j;
    __local unsigned char pipeinput[802816];
//    if ( filtersize == 1024 && op_size == 1024 )
//    {
//        if (tx == 0 && ty == 0)
//        {
//        
//            printf("Iam in 4pointwise layer\n");
//            printf("filtersize %d\n",filtersize);
//            printf("rows %d\n",rows);
//            printf("cols %d\n",cols);
//        }
//    }
    if (tx == 0 && ty == 0)
    {
        for ( i = 0; i < (rows*cols); i++ )
        {
            for ( j = 0; j < filtersize; j++ )
            {
                read_pipe_block(pipePath1, &pipeinput[(j*rows*cols)+i]);	            
                //printf("%d\t",pipeinput[(j*rows*cols)+i]);
            }
        }
//        for ( i = (112*112); i < (112*112*2); i++ )
//        {
//            printf("%d\t",pipeinput[i]);	
//        }
    } 
    
//    int pi;
//    if (tx == 0 && ty == 0)
//    {
//        for ( i = 0; i < 16; i++ )across tile travel
//        {
//            for ( pi = 0; pi < 16; pi++ )across tile travel
//            {
//                for (l_i = 0; l_i < localWorkSize; l_i++)Inside tile travel
//                {
//                    for (l_j = 0; l_j < localWorkSize; l_j++)Inside tile travel
//                    { 
//                        for ( j = 0; j < filtersize; j++ )
//                        {
//                            read_pipe_block(pipePath1, &pipeinput[(j*rows*cols)+( l_i*cols + l_j )+(pi*localWorkSize)+(i*localWorkSize*cols)]);		    
//                        }
//                    }
//                }
//            }
//        }
//    } 
	while (filter_count < op_size) {
		int output_shift = rows * cols * filter_count;
		
		for (i = 0; i < filtersize; i++,findex++) {
			sum += pipeinput[(ty * get_global_size(0) + tx) + (rows * cols * i)] * (filter_k[findex] - Z2); 
		}
		
		//sum = (int)((M * sum) + (bias[filter_count] * Sbias));
		sum = sum + bias[filter_count];
		sum = sum * (float)Q / (float)2147483648;
		sum = sum + ((right_shift < 1) ? 0 : (1 << (right_shift - 1)));
		sum = sum >> right_shift;

		/*if (tx == 4 && ty == 2) {
			//printf("M: %f\tbias: %f\t\n",M,Sbias);
			//printf("Summ: %d\t\n",(int)((M * sum) + (bias[filter_count] * Sbias)));
			printf(" Point Sum: %d\t\n",sum);
		}*/

		if (sum <= 0) {
			sum = 0;		
		} else if (sum >= 255) 
			sum = 255;
//		if (filtersize == 1024 && op_size == 1024)
//        {    
//		    output[(ty * get_global_size(0) + tx) + output_shift] = (unsigned char)sum;
//        }
//        else
//        {        
            pipeIp = (unsigned char)sum;
            write_pipe_block(pipePath0, &pipeIp);
//        }		
        sum = 0;
		filter_count++;
	}
}
__kernel void avgPool(__global unsigned char* output, 
					  __global unsigned char* inp_image, 
					  int rows, int cols ) {

        int tx = get_global_id(0);
        int sum = 0;
        int i,j;
	    int input_shift = rows * cols;
        __local unsigned char pipeinput[60176];
        if (tx == 0)
        {
            for ( i = 0; i < 49; i++ )
            {
                for ( j = 0; j < 1024; j++ )
                {
                    read_pipe_block(pipePath0, &pipeinput[(j*rows*cols)+i]);	            
                    //printf("%d\t",pipeinput[(j*rows*cols)+i]);
                }
            }
    //        for ( i = (112*112); i < (112*112*2); i++ )
    //        {
    //            printf("%d\t",pipeinput[i]);	
    //        }
        }
		for (i = 0; i < rows * cols; i++) {
			sum += pipeinput[i + ( tx * input_shift)];
		}
		/*{
			//printf("M: %f\tbias: %f\t\n",M,Sbias);
			//printf("Summ: %d\t\n",(int)((M * sum) + (bias[filter_count] * Sbias)));
			printf("Sum/49: %d\t\n",sum/49);
		}*/
		output[tx] = sum / 49;
}
