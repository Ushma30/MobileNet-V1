#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>

#define HEIGHT 224
#define WIDTH 224
#define K 3
#define K_P 1
#define FILTER_SIZE_L1 32
#define FILTER_SIZE_L3 64
#define FILTER_SIZE_L5 128
#define FILTER_SIZE_L9 256
#define FILTER_SIZE_L13 512
#define FILTER_SIZE_L25 1024
#define FILTER_SIZE_L29 1000


unsigned char image[HEIGHT * WIDTH * K]; //image with 3 input channels
int decode_image(unsigned char frame[HEIGHT * WIDTH * K], char filename[]);
void readSquezeNetKernel(unsigned char *m, int read_size) 
{

	FILE *fp;	
	char buff[255];
	double n;
	fp = fopen("weight.txt", "r");
	//int sizeInt = K * K * K * 32 *sizeof(int);
	int i=0;
	for(i = 1; i < read_size + 1; i++)
	{	
		fscanf(fp, "%s", buff);
		n = atof(buff);
		m[i-1] = n;
	}
	fclose(fp);
}
//Function to read image files in C
int decode_image(unsigned char frame[HEIGHT * WIDTH * K],char filename[])
{
	FILE *pFile;
	pFile = fopen(filename, "r"); //read mode
	fseek(pFile, 15, SEEK_SET);
	fread(frame, sizeof(unsigned char), HEIGHT * WIDTH * K, pFile);
	fclose(pFile);
	return 0;
}
//Function to load OpenCL kernel - taken from code given by T.A. Arnab 
long LoadOpenCLKernel(char const* path, char **buf)
{
	FILE  *fp;
	size_t fsz;
	long   off_end;
	int    rc;

	/* Open the file */
	fp = fopen(path, "r");
	if( NULL == fp ) {
		return -1L;
	}

	/* Seek to the end of the file */
	rc = fseek(fp, 0L, SEEK_END);
	if( 0 != rc ) {
		return -1L;
	}

	/* Byte offset to the end of the file (size) */
	if( 0 > (off_end = ftell(fp)) ) {
		return -1L;
	}
	fsz = (size_t)off_end;

	/* Allocate a buffer to hold the whole file */
	*buf = (char *) malloc( fsz+1);
	if( NULL == *buf ) {
		return -1L;
	}

	/* Rewind file pointer to start of file */
	rewind(fp);

	/* Slurp file into buffer */
	if( fsz != fread(*buf, 1, fsz, fp) ) {
		free(*buf);
		return -1L;
	}

	/* Close the file */
	if( EOF == fclose(fp) ) {
		free(*buf);
		return -1L;
	}


	/* Make sure the buffer is NUL-terminated, just in case */
	(*buf)[fsz] = '\0';

	/* Return the file size */
	return (long)fsz;
}
void display_data(unsigned char* data,int num) {
	int i,j;
	for (j = 0 ;j < num ; j++){
		for(i = 0; i < num; i++){
			printf("%d\t", data[j*WIDTH+i]);
		}
		printf("\n");
	}
	printf("\n");
}
//This is the main function
int main(int argc, char** argv) {

	//define memory for inputs and kernel
	unsigned char* filter = (unsigned char*) malloc(FILTER_SIZE_L25*FILTER_SIZE_L25*K*K*K*sizeof(unsigned char));
	unsigned char* image_r = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //R channel
	unsigned char* image_g = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //G channel
	unsigned char* image_b = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //B channel
	int i,j,k;
	
	int stride = 2;
	int op_size = 32;
	int err;
	cl_device_id device_id;             // compute device id 
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel;                   // compute kernel

	cl_mem d_image_r; //R channel
	cl_mem d_image_g; //G channel
	cl_mem d_image_b; //B channel

	cl_mem d_filter; //filter
	cl_mem d_output; //output image
	cl_event myevent; //profiling event
	cl_ulong start; //time start
	cl_ulong end; //time stop
	cl_float kernelExecTimeNs;

	unsigned int* output_image = (unsigned int*) malloc(FILTER_SIZE_L1 * (HEIGHT/2) * (WIDTH/2) * sizeof(unsigned int));
	
	printf("Initializing OpenCL device...\n"); 

	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);
	
	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	
	// Connect to a compute device
	int gpu = 1;
	err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}
  
	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command commands
	commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source file
	char *KernelSource;
	long lFileSize;

	lFileSize = LoadOpenCLKernel("kernel.cl", &KernelSource);
	if( lFileSize < 0L ) {
		perror("File read failed");
		return 1;
	}

	program = clCreateProgramWithSource(context, 1, (const char **) &KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "convolute", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	decode_image(image,"Cat_Image0.ppm"); //call the read function
    
	//separate R,G and B pixels
	int count = 0;

	for(i = 0; i<HEIGHT * WIDTH * K; i+=3)
	{
		image_r[count] = image[i];
		count++;
	}
	count = 0;

	for(j = 1; j<HEIGHT * WIDTH * K; j+=3)
	{
		image_g[count] = image[j]; 
		count++;
	}
	count = 0;
    
	for(k = 2; k<HEIGHT * WIDTH * K; k+=3)
	{
		image_b[count] = image[k];
		count++; 
	}

	printf("Image R\n");
	for (j = 0; j < 10; j++){
		for(i = 0; i < 10; i++){
			printf("%d\t", image_r[j*224+i]);
		}
		printf("\n");
	}
	printf("\n");
	printf("Image G\n");
	for (j = 0; j < 10; j++){
		for(i = 0; i < 10; i++){
			printf("%d\t", image_g[j*224+i]);
		}
		printf("\n");
	}
	printf("\n");
	printf("Image B\n");
	for (j = 0; j < 10; j++){
		for(i = 0; i < 10; i++){
			printf("%d\t", image_b[j*224+i]);
		}
		printf("\n");
	}
    printf("\n");
	//Get filter values
	readSquezeNetKernel(filter, (FILTER_SIZE_L1*K*K*K));

	printf("Filter Layer 1 values");
	for(i = 0; i < FILTER_SIZE_L1*K*K*K; i++){
		printf("%d\t", filter[i]);
	}

	//Create buffer for device
	d_image_r = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT*WIDTH*sizeof(unsigned char), image_r, &err);
	d_image_g = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT*WIDTH*sizeof(unsigned char), image_g, &err);
	d_image_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT*WIDTH*sizeof(unsigned char), image_b, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L1*sizeof(unsigned int), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L1*K*K*K*sizeof(unsigned char), filter, &err);

	if (!d_image_r || !d_image_g || !d_image_b || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_r, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_r, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands, d_image_g, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_g, 0, NULL, NULL);   
	err = clEnqueueWriteBuffer(commands, d_image_b, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_b, 0, NULL, NULL);   
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L1*K*K*K*sizeof(unsigned char), filter, 0, NULL, NULL);   
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	size_t localWorkSize[2], globalWorkSize[2];
	int rows = HEIGHT;
	int cols = WIDTH;
	int filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_r);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_image_g);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_image_b);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&filtersize);
    err |= clSetKernelArg(kernel, 8, sizeof(int), (void *)&stride);
    err |= clSetKernelArg(kernel, 9, sizeof(int), (void *)&op_size);

	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 112;
	globalWorkSize[1] = 112;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L1*(HEIGHT/2)*(WIDTH/2)*sizeof(unsigned int), output_image, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}
     
	//Get kernel execution time
	printf("Kernel Execution time for Layer 1: %f\n",kernelExecTimeNs/1000000000);
	
	for (j = 0; j < 5; j++){
		for(i = 0; i < 5; i++){
			printf("%d\t", output_image[(j*112+i) + (112 * 112 * 8)]);
		}
		printf("\n");
	}
    printf("\n");

	//Layer 2 Depth-Wise Convolution

	cl_mem d_image_l2;	//Layer 2 - Input Data
	kernelExecTimeNs = 0;
	op_size = 32;
	stride = 1;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L1*K*K));
	
	printf("Filter Layer 2 values");
	for(i = 0; i < FILTER_SIZE_L1*K*K; i++){
		printf("%d\t", filter[i]);
	}

	//Create buffer for device
	d_image_l2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L1*sizeof(unsigned int), output_image, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L1*sizeof(unsigned int), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L1*K*K*sizeof(unsigned char), filter, &err);	

	if (!d_image_l2 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l2, CL_TRUE, 0, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L1*sizeof(unsigned int), output_image, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L1*K*K*sizeof(unsigned char), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/2;
	cols = WIDTH/2;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l2);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 112;
	globalWorkSize[1] = 112;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;	
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L1*(HEIGHT/2)*(WIDTH/2)*sizeof(unsigned int), output_image, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	printf("Kernel Execution time for Layer 2: %f\n",kernelExecTimeNs/1000000000);

	for (j = 0; j < 5; j++){
		for(i = 0; i < 5; i++){
			printf("%d\t", output_image[(j*112+i) + (112 * 112 * 31)]);
		}
		printf("\n");
	}
    printf("\n");

	//Layer 3 Point-Wise Convolution
	
	unsigned int* output_image_l3 = (unsigned int*) malloc(FILTER_SIZE_L3 * (HEIGHT/2) * (WIDTH/2) * sizeof(unsigned int));

	cl_mem d_image_l3;	//Layer 3 - Input Data
	kernelExecTimeNs = 0;
	op_size = 64;

	readSquezeNetKernel(filter, (K_P*K_P*FILTER_SIZE_L1*FILTER_SIZE_L3));

	printf("Filter Layer 3 values");
	for(i = 0; i < K_P*K_P*FILTER_SIZE_L1*FILTER_SIZE_L3; i++){
		printf("%d\t", filter[i]);
	}

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	//Create buffer for device
	d_image_l3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L1*sizeof(unsigned int), output_image, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L3*sizeof(unsigned int), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K_P*K_P*FILTER_SIZE_L1*FILTER_SIZE_L3*sizeof(unsigned char), filter, &err);

	if (!d_image_l3 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l3, CL_TRUE, 0, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L1*sizeof(unsigned int), output_image, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, K_P*K_P*FILTER_SIZE_L1*FILTER_SIZE_L3*sizeof(unsigned char), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/2;
	cols = WIDTH/2;
	filtersize = FILTER_SIZE_L1;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l3);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
	
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 112;
	globalWorkSize[1] = 112;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L3*(HEIGHT/2)*(WIDTH/2)*sizeof(unsigned int), output_image_l3, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 3: %f\n",kernelExecTimeNs/1000000000);

	for (j = 0; j < 5; j++){
		for(i = 0; i < 5; i++){
			printf("%d\t", output_image_l3[(j*112+i)]);
		}
		printf("\n");
	}
    printf("\n");

	//Shutdown and cleanup
	free(image_r);
	free(image_g);
	free(image_b);
	free(filter);
	free(output_image);
	free(output_image_l3);/*free(output_image_l4);free(output_image_l5);
 	free(output_image_l5);free(output_image_l6);free(output_image_l7);
	free(output_image_l8);free(output_image_l9);free(output_image_l10);
	free(output_image_l11);free(output_image_l12);free(output_image_l13);
	free(output_image_l14);free(output_image_l15);free(output_image_l16);
	free(output_image_l17);free(output_image_l18);free(output_image_l19);
	free(output_image_l20);free(output_image_l21);free(output_image_l22);
	free(output_image_l23);free(output_image_l24);free(output_image_l25);
	free(output_image_l26);free(output_image_l27);free(output_image_l28);free(output_image_l29);free(output_softmax);*/
	clReleaseMemObject(d_image_r);
	clReleaseMemObject(d_image_g);
	clReleaseMemObject(d_image_b);
	clReleaseMemObject(d_image_l2);
	clReleaseMemObject(d_image_l3);
	/*clReleaseMemObject(d_image_l4);
	clReleaseMemObject(d_image_l5);	
	clReleaseMemObject(d_image_l6);
	clReleaseMemObject(d_image_l7);
	clReleaseMemObject(d_image_l8);
	clReleaseMemObject(d_image_l9);
	clReleaseMemObject(d_image_l10);
	clReleaseMemObject(d_image_l11);
	clReleaseMemObject(d_image_l12);clReleaseMemObject(d_image_l13);clReleaseMemObject(d_image_l14);clReleaseMemObject(d_image_l15);
	clReleaseMemObject(d_image_l16);clReleaseMemObject(d_image_l17);clReleaseMemObject(d_image_l18);clReleaseMemObject(d_image_l19);clReleaseMemObject(d_image_l20);
	clReleaseMemObject(d_image_l21);clReleaseMemObject(d_image_l22);clReleaseMemObject(d_image_l23);clReleaseMemObject(d_image_l24);clReleaseMemObject(d_image_l25);
	clReleaseMemObject(d_image_l26);clReleaseMemObject(d_image_l27);clReleaseMemObject(d_image_l28);clReleaseMemObject(d_image_l29);*/
	clReleaseMemObject(d_output);
	clReleaseMemObject(d_filter);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	return 0;
}
