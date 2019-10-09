#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>

struct Matrix {
	float* addr;
	int height;
	int width;
};

#define TILE_WIDTH 32
#define BLOCK_WIDTH 32

#define CHECK(call) {																	\
	const cudaError_t error = call;														\
	if (error != cudaSuccess) {															\
		printf("Error: %s:%d, ", __FILE__, __LINE__);									\
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));				\
		assert(false);																	\
	}																					\
}	

void read_args(char* arg_file, int* ppsi, int* pnum);
struct Matrix readMatrix(char* file_name);
void writeMatrix(char* fileName, Matrix M, bool numeric);
void writeIndexMatrix(char* file_name, Matrix M);
void writeNumericMatrix(char* file_name, Matrix M);
__global__ void calcMatrixDist(Matrix M, Matrix N, Matrix P); 
__global__ void findMinDistIdx(Matrix dist, Matrix ret, int psi, int num);


int main(int argc, char **argv) {
	assert(argc == 5);
	char *arg_file = argv[1], *model_file = argv[2], *input_file = argv[3], *output_file = argv[4];

	int psi, num;
	read_args(arg_file, &psi, &num);
	struct Matrix model = readMatrix(model_file);
	struct Matrix input = readMatrix(input_file);

	printf("Loading matrixs from %s and %s completed\n", model_file, input_file);
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
	
	float *deModel = NULL, *deInput = NULL, *deDist = NULL, *deRet = NULL;
	cudaMalloc((void **)&deModel, model.height*model.width*sizeof(float));
	cudaMalloc((void **)&deInput, input.height*input.width*sizeof(float));
	cudaMalloc((void **)&deDist, model.height*input.height*sizeof(float));
	cudaMalloc((void **)&deRet, input.height*num*sizeof(int));

	cudaMemcpy(deModel, model.addr, model.height*model.width*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deInput, input.addr, input.height*input.width*sizeof(float), cudaMemcpyHostToDevice);
	
	free(model.addr);
	free(input.addr);
	model.addr = deModel;
	input.addr = deInput;

	struct Matrix dist;
	dist.height = input.height;
	dist.width = model.height;
	dist.addr = deDist;
	
	int dx = (int)(ceil((float)model.height/TILE_WIDTH)),
		dy = (int)(ceil((float)input.height/TILE_WIDTH));
	dim3 dimGrid(dx, dy);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	printf("Calculating distance matrix...... Grid: (%d, %d), Block: (%d,%d)\n", dx, dy, TILE_WIDTH, TILE_WIDTH);
	calcMatrixDist<<<dimGrid, dimBlock>>>(input, model, dist);

	/*
		输出中间结果-距离矩阵，验证calcMatrixDist正确性
	*/
	/*
	struct Matrix toWrite;
	toWrite.height = dist.height;
	toWrite.width = model.height;
	toWrite.addr = (float*)malloc(toWrite.height*toWrite.width*sizeof(float));
	cudaMemcpy(toWrite.addr, deDist, toWrite.height*toWrite.width*sizeof(float), cudaMemcpyDeviceToHost);
	writeNumericMatrix("bin/dist.csv", toWrite);
	free(toWrite.addr);
	*/

	struct Matrix ret;
	ret.height = input.height;
	ret.width = num;
	ret.addr = deRet;
	int bx = (int)(ceil((float)ret.height/BLOCK_WIDTH));
	printf("Finding index with minimum distance...... Grid: %d, Block: %d\n", bx, num);
	findMinDistIdx<<<bx, num>>>(dist, ret, psi, num);

	ret.addr = (float*)malloc(ret.height*num*sizeof(float));
	cudaMemcpy(ret.addr, deRet, ret.height*num*sizeof(float), cudaMemcpyDeviceToHost);
	writeIndexMatrix(output_file, ret);
	free(ret.addr);

	cudaFree(deModel);
	cudaFree(deInput);
	cudaFree(deDist);
	cudaFree(deRet);
	cudaDeviceReset();
	return 0;
}


void read_args(char* arg_file, int* ppsi, int* pnum) {
	FILE* f = fopen(arg_file, "r");
	if (f == NULL) {
		printf("Fail to read args\n");
		assert(false);
	}
	assert(fscanf(f, "%d %d", ppsi, pnum) >= 0);
}


struct Matrix readMatrix(char* matrix_file) {
	struct Matrix matrix;

	FILE* f = fopen(matrix_file, "r");
	if (f == NULL) {
		printf("Fail to read model\n");
		assert(false);
	}

	int height, width;
	assert(fscanf(f, "%d\t%d", &height, &width) >= 0);
	matrix.addr = (float*)malloc(height*width*sizeof(float));
	matrix.height = height;
	matrix.width = width;

	float* ptr = matrix.addr;
	for (int i = 0; i < height*width; ++i, ++ptr)
		assert(fscanf(f, "%f", ptr) >= 0);

	return matrix;
}


void writeMatrix(char* file_name, Matrix M, bool numeric) {
	FILE* f = fopen(file_name, "w");
	if (f == NULL) {
		printf("Fail to write result\n");
		assert(false);
	}

	fprintf(f, "%d\t%d\n", M.height, M.width);
	float* ptr = M.addr;
	for (int i = 0; i < M.height; ++i) {
		for (int j = 0; j < M.width; ++j, ++ptr) {
			if (numeric)
				fprintf(f, "%f\t", *ptr);
			else
				fprintf(f, "%d\t", (int)(*ptr));
		}
		fprintf(f, "\n");
	}
}

void writeNumericMatrix(char* fileName, Matrix M) {
	writeMatrix(fileName, M, true);
}

void writeIndexMatrix(char* fileName, Matrix M) {
	writeMatrix(fileName, M, false);
}


__global__ void calcMatrixDist(Matrix M, Matrix N, Matrix P) {
	assert(M.width == N.width);
	assert(P.height == M.height && P.width == N.height);

	__shared__ float sharedM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sharedN[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	float v = 0;

	int rowM = by*TILE_WIDTH + ty, rowN = bx*TILE_WIDTH + ty;
	for (int i = 0; i < (int)(ceil((float)M.width/TILE_WIDTH)); ++i) {
		int offM = rowM*M.width + i*TILE_WIDTH + tx;
		if (i*TILE_WIDTH + tx < M.width && rowM < M.height)
			sharedM[ty][tx] = M.addr[offM];
		else
			sharedM[ty][tx] = 0;
		
		int offN = rowN*N.width + i*TILE_WIDTH + tx;
		if (i*TILE_WIDTH + tx < N.width && rowN < N.height)
			sharedN[ty][tx] = N.addr[offN];
		else
			sharedN[ty][tx] = 0;

		__syncthreads();

		for (int j = 0; j < TILE_WIDTH; ++j)
			v += (sharedM[ty][j] - sharedN[tx][j])*(sharedM[ty][j] - sharedN[tx][j]);
		
		__syncthreads();
	}

	int row = by*TILE_WIDTH + ty, col = bx*TILE_WIDTH + tx;
	if (row < P.height && col < P.width)
		P.addr[row*P.width + col] = v;
	/*	
	printf("block: (%d,%d)  thread: (%d,%d), write into: (%d, %d): %f\n", 
			bx, by, tx, ty, row, col, v);
	*/
}


__global__ void findMinDistIdx(Matrix dist, Matrix ret, int psi, int num) {
	assert(dist.width == psi*num);
	assert(ret.height == dist.height);
	assert(ret.width == num);
	int bx = blockIdx.x, tx = threadIdx.x;

	for (int i = 0; i < BLOCK_WIDTH; ++i) {
		int row = bx*BLOCK_WIDTH + i;
		if (row >= ret.height)
			break;
		int offset = row*dist.width + tx*psi, idx = 0;
		double min = dist.addr[offset];
		for (int j = 1; j < psi; ++j) {
			if (dist.addr[offset + j] < min) {
				min = dist.addr[offset + j];
				idx = j;
			}
		}
		ret.addr[row*ret.width + tx] = idx;
	}
}

