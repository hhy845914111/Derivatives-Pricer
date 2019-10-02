//	Example of Multi-threading in C++ on Windows operating system
// include necessary header files
#include <malloc.h>

#define MAX_NUM_THREADS 8
using namespace std;

/* prototype */
unsigned __stdcall RunThread(void *param);
int MyParallelFunction(int jThread);

/* Global Variables */
//  Data and parameters required by functions running on multiple thread
int nThreads;
double Data, *Params;
//  ........
//  Results from each thread need to be stored in global memory
double *Result;

HANDLE threads[MAX_NUM_THREADS];
FILE *fout;

/* create thread argument struct for Run_Thread() */
typedef struct _thread_data_t {
	int tid;
	//	double stuff;
} thread_data_t;

int _tmain(int argc, _TCHAR* argv[])
{
	int i, j, nThreads;
	double *A, *B, x, y;

	unsigned threadID[MAX_NUM_THREADS];
	// create a thread_data_t argument array 
	thread_data_t thr_data[MAX_NUM_THREADS];

	//  code to set everything up

	//  use malloc to allocate dynamic memory



	//	Set up multi-threading here

	for (i = 0; i < nThreads; i++) {
		thr_data[i].tid = i;
		threads[i] = (HANDLE)_beginthreadex(NULL, 0, RunThread, &thr_data[i], 0, &threadID[i]);
	}
	WaitForMultipleObjects(nThreads, threads, TRUE, INFINITE);
	for (i = 0; i < nThreads; i++) CloseHandle(threads[i]);



	//	code to work with Results from parallel threads


	fclose(fout);
	//   free the work arrays where dynamic memory has been allocated - no memory leaks
	free(A);
	free(B);
	free(Params);
	free(Results);

	return 0;

}

unsigned __stdcall RunThread(void *param)
{
	int icheck, iThread;
	thread_data_t *data = (thread_data_t *)param;
	iThread = data->tid;
	icheck = MyParallelFunction(iThread);
	return 1;
}

int MyParallelFunction(int jThread)
{
	//	Code to run in parallel on multiple threads
	//  jThread identifies the thread number
	//	Variables can be defined and memory can be allocated inside this function
	//  The function runs indepednently on each thread with its own memory
	//  There is access to global memry

	int i, j;
	double *x, *y, *SimulatedValues;


	//	Code to run on thread here
	//  Monte Carlo simulations, finite difference algorithm, ....


	return 1;

}
