
__kernel void convolute(__global unsigned int* output, __global unsigned char* inp_image_r, __global unsigned char* inp_image_g, __global unsigned char* inp_image_b, 
						__global int* filter_k, int rows, int cols, int filtersize, int stride, int op_size ) {

	int tx = get_global_id(0);
	int ty = get_global_id(1);
	
	int half_filtersize = (filtersize)/2;

	int sum = 0;
	int xindex=0, yindex=0, findex=0, filter_count=0;
	int i,j,l;
	while (filter_count < op_size) {
		int output_shift = (rows / 2) * (cols / 2) * filter_count;
		filter_count++;
		for(i = -half_filtersize; i<= half_filtersize; i++){
			yindex = ty * stride + i;
			for(j = -half_filtersize; j<= half_filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex < 0 || xindex < 0) {
					sum +=  0 * filter_k[findex];
				}
				else {
 					sum +=  inp_image_r[yindex * get_global_size(0) * stride + xindex] * filter_k[findex];
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
 					sum +=  inp_image_g[yindex * get_global_size(0) * stride + xindex] * filter_k[findex];
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
 					sum +=  inp_image_b[yindex * get_global_size(0) * stride + xindex] * filter_k[findex];
				}
			}
		}
		if (sum <= 0) {
			sum = 0;		
		}
		//if ()
		output[(ty * get_global_size(0) + tx) + output_shift] = sum;
		//}
	}

}
