
__kernel void convolute(__global unsigned char* output, 
						__global unsigned char* inp_image_r, 
						__global unsigned char* inp_image_g, 
						__global unsigned char* inp_image_b, 
						__global unsigned char* filter_k,
						__global int* bias,
						int rows, int cols, int filtersize, int stride, int op_size,
						float M, float Sbias, unsigned char Z1, unsigned char Z2 ) {

	int tx = get_global_id(0);
	int ty = get_global_id(1);
	
	int half_filtersize = (filtersize)/2;

	int sum = 0;
	int xindex=0, yindex=0, findex=0, filter_count=0;
	int i,j,l;
	while (filter_count < op_size) {
	
		int output_shift = (rows / 2) * (cols / 2) * filter_count;
		
		for(i = -half_filtersize; i<= half_filtersize; i++){
			yindex = ty * stride + i;
			for(j = -half_filtersize; j<= half_filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex < 0 || xindex < 0) {
					sum +=  0 * filter_k[findex];
				}
				else {
					/*if (tx == 4 && ty == 2 && filter_count == 0) {
						printf("Image r: %d\t%d\n",(inp_image_r[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2),filter_k[findex]);
					}*/
 						sum +=  (inp_image_r[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2);
				}
			}
		}
		for(i = -half_filtersize; i<= half_filtersize; i++){
			yindex = ty * stride + i;
			for(j = -half_filtersize; j<= half_filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex < 0 || xindex < 0) {
					sum +=  0 * filter_k[findex];
				}
				else {
					/*if (tx == 4 && ty == 2 && filter_count == 0) {
						printf("Img g: %d\t%d\n",(inp_image_g[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2),filter_k[findex]);
					}*/
 					sum +=  (inp_image_g[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2);
				}
			}
		}
		for(i = -half_filtersize; i<= half_filtersize; i++){
			yindex = ty * stride + i;
			for(j = -half_filtersize; j<= half_filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex < 0 || xindex < 0) {
					sum +=  0 * filter_k[findex];
				}
				else {
					/*if (tx == 4 && ty == 2 && filter_count == 0) {
						printf("Img b: %d\t%d\n",(inp_image_b[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2),filter_k[findex]);
					}*/
 					sum +=  (inp_image_b[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2);
				}
			}
		}
		
		/*if (tx == 4 && ty == 2 && filter_count == 0) {
			printf("M: %f\tbias: %f\t\n",M,Sbias);
			printf("Summ: %d\t\n",(int)((M * sum) + (bias[filter_count] * Sbias)));
			printf("Sum: %d\t\n",sum);
		}*/
		sum = (int)((M * sum) + (bias[filter_count] * Sbias));

		if (sum <= 0) {
			sum = 0;		
		}
		
		output[(ty * get_global_size(0) + tx) + output_shift] = sum;
		sum = 0;
		filter_count++;
	}
}

__kernel void depthwise(__global unsigned char* output, 
						__global unsigned char* inp_image, 
						__global unsigned char* filter_k, 
						__global int* bias, 
						int rows, int cols, int filtersize, int stride, int op_size,
						float M, float Sbias, unsigned char Z2 ) { 

	int tx = get_global_id(0);
	int ty = get_global_id(1);

	int half_filtersize = (filtersize)/2;

	int sum = 0;
	int xindex=0, yindex=0, findex=0, filter_count=0;
	int i,j,l;
	while (filter_count < op_size) {
		int output_shift = rows * cols * filter_count;
		
		for(i = -half_filtersize; i<= half_filtersize; i++){
			yindex = ty * stride + i;
			for(j = -half_filtersize; j<= half_filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex < 0 || xindex < 0) {
					sum +=  0 * filter_k[findex];
				}
				else {
 					sum +=  inp_image[yindex * get_global_size(0) * stride + xindex] * (filter_k[findex] - Z2);
				}
			}
		}

		sum = (int)((M * sum) + (bias[filter_count] * Sbias));

		if (sum <= 0) {
			sum = 0;		
		}

		output[(ty * get_global_size(0) + tx) + output_shift] = sum;
		sum = 0;
		filter_count++;	
	}
}

__kernel void pointwise(__global unsigned char* output, 
						__global unsigned char* inp_image, 
						__global unsigned char* filter_k, 
						__global int* bias, 
						int rows, int cols, int filtersize, int op_size,
						float M, float Sbias, unsigned char Z2 ) {  

	int tx = get_global_id(0);
	int ty = get_global_id(1);

	unsigned int sum = 0;
	int findex=0, filter_count=0;
	int i,j,l;
	while (filter_count < op_size) {
		int output_shift = rows * cols * filter_count;
		
		for (i = 0; i < filtersize; i++,findex++) {
			sum += inp_image[(ty * get_global_size(0) + tx) + (rows * cols * i)] * (filter_k[findex] - Z2); 
		}
		
		sum = (int)((M * sum) + (bias[filter_count] * Sbias));

		if (sum <= 0) {
			sum = 0;		
		}
		output[(ty * get_global_size(0) + tx) + output_shift] = sum;
		sum = 0;
		filter_count++;
	}
}
__kernel void avgPool(__global unsigned char* output, 
					  __global unsigned char* inp_image, 
					  int rows, int cols, int poolSize, int op_size ) {

        int tx = get_global_id(0);
        int sum = 0;
        int poolCnt=0;
        int i,j,l;
	    int input_shift = rows * cols;
		for (i = 0; i < poolSize * poolSize; i++) {
			sum += inp_image[i + ( tx * input_shift)];
		}
		output[tx] = sum / 49;
}