/****************************************************************************
 *               University of North Carolina Charlotte                     *
 *                        MobileNet V1 CNN                                  *
 *                        				                                    *
 *                                                                          *
 *                                                                          *
 *   Author:    1. Kaustubh Manohar Mhatre                                  *
 *              2. Ushma Bharucha                                           *
 *   Date: 08 June 2019														*
 ****************************************************************************/

/****************************************************************************
* Includes																	*
*****************************************************************************/
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
#include "layerdef.h"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

unsigned char image[HEIGHT_0 * WIDTH_0 * FDIM]; //image with 3 input channels
unsigned char* filter;
int err;

cl_device_id device_id;             // compute device id 
cl_context context;                 // compute context
cl_command_queue commands;          // compute command queue
cl_program program;                 // compute program
cl_kernel standard_conv;            // compute kernel for standard convolution
cl_kernel depthwise_conv;            // compute kernel for depthwise convolution
cl_kernel pointwise_conv;            // compute kernel for pointwise convolution

cl_mem d_filter; //filter
cl_mem d_output; //output image
cl_event myevent; //profiling event
cl_ulong start; //time start
cl_ulong end; //time stop
cl_float kernelExecTimeNs;
cl_uint dev_cnt = 0;
cl_platform_id platform_ids[100];


int decode_image(unsigned char frame[HEIGHT_0 * WIDTH_0 * FDIM], char filename[]);

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

int openClDeviceConfig(){

	printf("Initializing OpenCL device...\n"); 

	clGetPlatformIDs(0, 0, &dev_cnt);
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	
	// Connect to a compute device
	int gpu = 1;
	err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

}

int openClCreateContext() {
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
}

int openClCreateKernel() {
	
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

	// Create the compute kernel for standard convolution
	standard_conv = clCreateKernel(program, "convolute", &err);
	if (!standard_conv || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Create the compute kernel for depthwise convolution
	depthwise_conv = clCreateKernel(program, "depthwise", &err);
	if (!depthwise_conv || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Create the compute kernel for standard convolution
	pointwise_conv = clCreateKernel(program, "pointwise", &err);
	if (!pointwise_conv || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}
}

void seperateChannels(unsigned char* imd,unsigned char* im1,unsigned char* im2,unsigned char* im3){
    int i,j;    
    for(i=0,j=0; i<HEIGHT_0*WIDTH_0; i++,j+=3){
        im1[i] = imd[j];
        im2[i] = imd[j+1];
        im3[i] = imd[j+2];                
    }
}

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
/**
 * @brief  Get the weights from the numpy array file
 * @author  Kausutbh
 * @date June 7, 2019
 * @param 1. unsigned char* f : variable to put weights into
 *        2. char filename[] : File name of the weights filename
 *        3. int size
 * @return None
 */
void getWeights(unsigned char* f, char filename[], int size)
{
    FILE *latfile;
    latfile=fopen(filename,"r");
    /* 80 is the offset of numpy array file*/
    fseek(latfile, 80, SEEK_SET);
    fread(f,sizeof(unsigned char),size,latfile);
    fclose(latfile);
}
//Function to read image files in C
int decode_image(unsigned char frame[HEIGHT_0 * WIDTH_0 * FDIM],char filename[])
{
	FILE *pFile;
	pFile = fopen(filename, "r"); //read mode
	fseek(pFile, 15, SEEK_SET);
	fread(frame, sizeof(unsigned char), HEIGHT_0 * WIDTH_0 * FDIM, pFile);
	fclose(pFile);
	return 0;
}
//Function to load OpenCL kernel - taken from code given by T.A. Arnab 


void display_data(unsigned char* data,int num) {
	int i,j;
	for (j = 0 ;j < num ; j++){
		for(i = 0; i < num; i++){
			printf("%d\t", data[j*WIDTH_0+i]);
		}
		printf("\n");
	}
	printf("\n");
}

void convStandard (unsigned char* opfm) {

	cl_mem d_image_r; //R channel
	cl_mem d_image_g; //G channel
	cl_mem d_image_b; //B channel

	unsigned char* image_r = (unsigned char*) malloc(HEIGHT_0 * WIDTH_0 * sizeof(unsigned char)); //R channel
	unsigned char* image_g = (unsigned char*) malloc(HEIGHT_0 * WIDTH_0 * sizeof(unsigned char)); //G channel
	unsigned char* image_b = (unsigned char*) malloc(HEIGHT_0 * WIDTH_0 * sizeof(unsigned char)); //B channel

	int i,j,k;

	int bias[32] = {-7254, 13465, -1591, -2488, 9901, 13359, 1947, 16203, -2165, -7399, -5250, 13549, 17637, 9441, 3877, 17663, -7985, 13809, 12408, 11717, 
	-3441, -104, -13034, 17888, -12487, 15548, 26765, -2599, 14359, 10137, -2149, 20334};
	
	//Read pixel values from input image
	decode_image(image,"Cat_Image0.ppm"); 

	//separate R,G and B pixels
	seperateChannels(image, image_r, image_g, image_b);

	//Get filter values
    getWeights(filter,"weights/Conv2d_0",(IP_FM_1*FDIM*FDIM*FDIM));

	/* for (k = 0; k < IP_FM_1*FDIM*FDIM*FDIM; k++){
		printf("Filter [%d] %d\t",k,filter[k]);
	}
	printf("\n");*/

	//Create buffer for device
	d_image_r = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT_0*WIDTH_0*sizeof(unsigned char), image_r, &err);
	d_image_g = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT_0*WIDTH_0*sizeof(unsigned char), image_g, &err);
	d_image_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT_0*WIDTH_0*sizeof(unsigned char), image_b, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT_1)*(WIDTH_1)*IP_FM_1*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, IP_FM_1*FDIM*FDIM*FDIM*sizeof(unsigned char), filter, &err);

	if (!d_image_r || !d_image_g || !d_image_b || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_r, CL_TRUE, 0, HEIGHT_0*WIDTH_0*sizeof(unsigned char), image_r, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands, d_image_g, CL_TRUE, 0, HEIGHT_0*WIDTH_0*sizeof(unsigned char), image_g, 0, NULL, NULL);   
	err = clEnqueueWriteBuffer(commands, d_image_b, CL_TRUE, 0, HEIGHT_0*WIDTH_0*sizeof(unsigned char), image_b, 0, NULL, NULL);   
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, IP_FM_1*FDIM*FDIM*FDIM*sizeof(unsigned char), filter, 0, NULL, NULL);   
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	int rows = HEIGHT_0;
	int cols = WIDTH_0;
	int filtersize = FDIM;
	int no_fm_0 = OP_FM_0;
    int stride = 2;

	err = clSetKernelArg(standard_conv, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(standard_conv, 1, sizeof(cl_mem), (void *)&d_image_r);
	err |= clSetKernelArg(standard_conv, 2, sizeof(cl_mem), (void *)&d_image_g);
	err |= clSetKernelArg(standard_conv, 3, sizeof(cl_mem), (void *)&d_image_b);
	err |= clSetKernelArg(standard_conv, 4, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(standard_conv, 5, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(standard_conv, 6, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(standard_conv, 7, sizeof(int), (void *)&filtersize);
    err |= clSetKernelArg(standard_conv, 8, sizeof(int), (void *)&stride);
    err |= clSetKernelArg(standard_conv, 9, sizeof(int), (void *)&no_fm_0);

	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}
	
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 112;
	globalWorkSize[1] = 112;
	err = clEnqueueNDRangeKernel(commands, standard_conv, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
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
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, IP_FM_1*(HEIGHT_1)*(WIDTH_1)*sizeof(unsigned char), opfm, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}
     
	//Get kernel execution time
	printf("Kernel Execution time for Layer 0: %f\n",kernelExecTimeNs/1000000000);
	
	/* printf("Layer 1 Data\n"); 

	for (k = 0; k < 32; k++){
		for (j = 0; j < 112; j++){
			for(i = 0; i < 112; i++){
				printf("%u\t", opfm[(j*112+i) + k]);
			}
			printf("\n");
		}
    printf("\n");
	} */

	free(image_r);
	free(image_g);
	free(image_b);

	clReleaseMemObject(d_image_r);
	clReleaseMemObject(d_image_g);
	clReleaseMemObject(d_image_b);

}

void convDepthwise(unsigned char* ipfm, unsigned char* opfm, char* fileName, int iph, int ipw, int oph, int opw, int ip_fsize, int op_fsize, int stride) {
	
	cl_mem d_input;	//Input Data
	kernelExecTimeNs = 0;
	//int op_fm_1 = 32;
	int i,j,k;

	//Get filter values
	getWeights(filter,fileName,(ip_fsize*FDIM*FDIM));
	
	//Create buffer for device
	d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, iph*ipw*ip_fsize*sizeof(unsigned char), ipfm, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, oph*opw*op_fsize*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ip_fsize*FDIM*FDIM*sizeof(unsigned char), filter, &err);	

	if (!d_input || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_input, CL_TRUE, 0, iph*ipw*ip_fsize*sizeof(unsigned char), ipfm, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, ip_fsize*FDIM*FDIM*sizeof(unsigned char), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	int rows = iph;
	int cols = ipw;
	int filtersize = FDIM;
    
	err = clSetKernelArg(depthwise_conv, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(depthwise_conv, 1, sizeof(cl_mem), (void *)&d_input);
	err |= clSetKernelArg(depthwise_conv, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(depthwise_conv, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(depthwise_conv, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(depthwise_conv, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(depthwise_conv, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(depthwise_conv, 7, sizeof(int), (void *)&op_fsize);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = opw;
	globalWorkSize[1] = oph;
	err = clEnqueueNDRangeKernel(commands, depthwise_conv, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
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
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, op_fsize*oph*opw*sizeof(unsigned char), opfm, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	printf("Kernel Execution time for Layer 1: %f\n",kernelExecTimeNs/1000000000);
	/*
	printf("Data for Layer 2\n");
	int x = 0;
	for (k = 0; k < op_fsize; k++){
		printf("Feature Map %d \n", k);
		for (j = 0; j < oph; j++){
			for(i = 0; i < opw; i++){
				printf("%u\t", opfm[(j*opw+i) + (k*opw*oph)]);
				if (opfm[(j*opw+i) + (k*opw*oph)] > x) {
					x = opfm[(j*opw+i) + (k*opw*oph)];
				}
			}
			printf("\n");
		}
    	
	}
	
	printf("Max value %u\n",x);*/
	clReleaseMemObject(d_input);

}

void convPointwise(unsigned char* ipfm, unsigned char* opfm, char* fileName, int iph, int ipw, int oph, int opw, int ip_fsize, int op_fsize, int stride) {

	cl_mem d_input;	//Input Data
	kernelExecTimeNs = 0;
	int i,j,k;

	//Get filter values
	getWeights(filter,fileName,(ip_fsize*op_fsize*FDIM_P*FDIM_P));
	
	//Create buffer for device
	d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, iph*ipw*ip_fsize*sizeof(unsigned char), ipfm, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, oph*opw*op_fsize*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ip_fsize*op_fsize*FDIM_P*sizeof(unsigned char), filter, &err);	

	if (!d_input || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}
	
	err = clEnqueueWriteBuffer(commands, d_input, CL_TRUE, 0, iph*ipw*ip_fsize*sizeof(unsigned char), ipfm, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, ip_fsize*op_fsize*FDIM_P*FDIM_P*sizeof(unsigned char), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	int rows = iph;
	int cols = ipw;
	int filtersize = ip_fsize;
    
	err = clSetKernelArg(pointwise_conv, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(pointwise_conv, 1, sizeof(cl_mem), (void *)&d_input);
	err |= clSetKernelArg(pointwise_conv, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(pointwise_conv, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(pointwise_conv, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(pointwise_conv, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(pointwise_conv, 6, sizeof(int), (void *)&op_fsize);
	
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = opw;
	globalWorkSize[1] = oph;
	err = clEnqueueNDRangeKernel(commands, pointwise_conv, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
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
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, op_fsize*oph*opw*sizeof(unsigned char), opfm, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 2: %f\n",kernelExecTimeNs/1000000000);

	printf("Data for Layer 3\n");

	int x = 0;
	/* for (k = 0; k < 32; k++){
		for (j = 0; j < 112; j++){
			for(i = 0; i < 112; i++){
				printf("%u\t", opfm[(j*112+i) + k]);
				if (opfm[(j*112+i) + k] > x) {
					x = opfm[(j*112+i) + k];
				}
			}
			printf("\n");
		}
    	printf("\n");
	}
	printf("Max value %d\n",x);*/
	clReleaseMemObject(d_input);
}


//This is the main function
int main(int argc, char** argv) {

	filter = (unsigned char*) malloc(FILTER_MAX*FILTER_MAX*FDIM*FDIM*FDIM*sizeof(unsigned char));
	unsigned char* op_fm_0 = (unsigned char*) malloc(IP_FM_1 * HEIGHT_1 * WIDTH_1 * sizeof(unsigned char)); //output feature map for layer 0
	int i,j,k;

	openClDeviceConfig();
	openClCreateContext();
	openClCreateKernel();
	convStandard(op_fm_0); //Layer 0 - Standard Convolution
	
	//Layer 1 Depth-Wise Convolution

	unsigned char* op_fm_1 = (unsigned char*) malloc(IP_FM_2 * HEIGHT_2 * WIDTH_2 * sizeof(unsigned char)); //output feature map for layer 1
	convDepthwise(op_fm_0,op_fm_1, "weights/Conv2d_1_depthwise", HEIGHT_1, WIDTH_1, HEIGHT_2, WIDTH_2, IP_FM_1, IP_FM_2, 1);	//Layer 1 Depth-Wise Convolution
	
	//Layer 2 Point-Wise Convolution

	unsigned char* op_fm_2 = (unsigned char*) malloc(IP_FM_3 * HEIGHT_3 * WIDTH_3 * sizeof(unsigned char));	//output feature map for layer 2
	convPointwise(op_fm_1,op_fm_2, "weights/Conv2d_1_pointwise", HEIGHT_2, WIDTH_2, HEIGHT_3, WIDTH_3, IP_FM_2, IP_FM_3, 1);	//Layer 2 Point-Wise Convolution

	//Layer 3 Depth-Wise Convolution

	unsigned char* op_fm_3 = (unsigned char*) malloc(IP_FM_4 * HEIGHT_4 * WIDTH_4 * sizeof(unsigned char)); //output feature map for layer 2
	convDepthwise(op_fm_2,op_fm_3, "weights/Conv2d_2_depthwise", HEIGHT_3, WIDTH_3, HEIGHT_4, WIDTH_4, IP_FM_3, IP_FM_4, 2);	//Layer 3 Depth-Wise Convolution

	//Shutdown and cleanup
	free(filter);
	free(op_fm_0);
	free(op_fm_1);
	free(op_fm_2);
	clReleaseMemObject(d_output);
	clReleaseMemObject(d_filter);
	clReleaseProgram(program);
	clReleaseKernel(standard_conv);
	clReleaseKernel(depthwise_conv);
	clReleaseKernel(pointwise_conv);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	return 0;
}
