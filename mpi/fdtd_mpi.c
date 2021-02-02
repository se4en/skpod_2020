#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define TMAX 1000
#define NX 2000
#define NY 2600

double bench_t_start, bench_t_end;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock ();
}

void bench_timer_stop()
{
  bench_t_end = rtclock ();
}

void bench_timer_print()
{
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

//----------------------------------------------------------------------------------------------------------------------------------------------------

static
void init_array (int tmax,
   int nx,
   int ny,
   float ex[ nx][ny],
   float ey[ nx][ny],
   float hz[ nx][ny],
   float _fict_[ tmax])
{
  int i, j;
  for (i = 0; i < tmax; i++)
    _fict_[i] = (float) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
        ex[i][j] = ((float) i*(j+1)) / nx;
        ey[i][j] = ((float) i*(j+2)) / ny;
        hz[i][j] = ((float) i*(j+3)) / nx;
      }
}

static
void kernel_fdtd_2d(int tmax,
      int nx,
      int ny,
      float ex[nx][ny],
      float ey[nx][ny],
      float hz[nx][ny],
      float _fict_[tmax],
      int myid, 
      int numprocs)
{
  int start_elem, end_elem, elem_num;
  int start_row, end_row, row_num;
  int r_n, st_row;
  int i, j, t;
  float *buffer;
  MPI_Status status;
  MPI_Request request;

  for(t = 0; t < tmax; ++t)
        {
        // not parallel
        if (myid==0) {
            for (j = 0; j < ny; ++j) 
                ey[0][j] = _fict_[t]; 
        }

        // preparation
        int k = nx/numprocs;
        if (myid==numprocs-1) 
            row_num = k + nx%numprocs;
        else
            row_num = k;
        start_row = k * myid;
        end_row = start_row + row_num;


        // update E
        // send matrix
        if (myid == 0) {
            for (int i = 1; i < numprocs; i++) {
                st_row=k*i;
                if (i==numprocs-1)
                   r_n = k + nx%numprocs;
                else  
                   r_n = k;
        	    MPI_Isend(&ey[st_row*ny], r_n*ny, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &request);
            }
        }
        // get rows
        if (myid == 0) {
            buffer = &ey[(start_row +1) * ny][0];
        } 
        else {
            buffer = (float *) malloc((end_row - start_row)*ny*sizeof(float));
            MPI_Recv(buffer, (end_row - start_row) * ny, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        }
        // update rows of E for each process
        for (i = 0; i < end_row-start_row; ++i) 
            for (j = 0; j < ny; ++j)
                buffer[(i+start_row)*ny + j] = buffer[(i+start_row)*ny + j] - 0.5f*( hz[(start_row+i)*ny + j] - hz[(start_row+i-1)*ny + j] );
        // send result
        if (myid > 0) 
            MPI_Isend(buffer, (end_row - start_row)*ny, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, &request);
        // get result
        if (myid == 0) {
            for (int i = 1; i < numprocs; i++) {
                st_row=k*i;
                if (i==numprocs-1)
                    r_n=k+nx % numprocs;
                else 
                    r_n=k;
                MPI_Recv(&ey[st_row * ny], r_n*ny, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);
            }
        }
        else {
            free(buffer);
        }


        // update E
        // send matrix
        if (myid == 0) {
            for (int i = 1; i < numprocs; i++) {
                st_row=k*i;
                if (i==numprocs-1)
                   r_n = k + nx%numprocs;
                else  
                   r_n = k;
        	    MPI_Isend(&ex[st_row*ny], r_n*ny, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &request);
            }
        }
        // get rows
        if (myid == 0) {
            buffer = &ex[start_row * ny][0];
        } 
        else {
            buffer = (float *) malloc((end_row - start_row)*ny*sizeof(float));
            MPI_Recv(buffer, (end_row - start_row) * ny, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        }
        // update rows of E for each process
        for (i = 0; i < end_row-start_row; ++i) 
            for (j = 1; j < ny; ++j)
                buffer[(i+start_row)*ny + j] = buffer[(i+start_row)*ny + j] - 0.5f*( hz[(start_row+i)*ny + j] - hz[(start_row+i)*ny + j-1] );
        // send result
        if (myid > 0) 
            MPI_Isend(buffer, (end_row - start_row)*ny, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, &request);
        // get result
        if (myid == 0) {
            for (int i = 1; i < numprocs; i++) {
                st_row=k*i;
                if (i==numprocs-1)
                    r_n=k+nx % numprocs;
                else 
                    r_n=k;
                MPI_Recv(&ex[st_row * ny], r_n*ny, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);
            }
        }
        else {
            free(buffer);
        }


        // update H
        // send matrix
        if (myid == 0) {
            for (int i = 1; i < numprocs; i++) {
                st_row=k*i;
                if (i==numprocs-1) {
                   r_n = k + nx%numprocs - 1;
        	        MPI_Isend(&hz[st_row*ny], r_n*ny, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &request);
                }
                else {  
                   r_n = k;
        	        MPI_Isend(&hz[st_row*ny], r_n*ny, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &request);
                }
            }
        }
        // get rows
        if (myid == 0) {
            buffer = &hz[start_row * ny][0];
        } 
        else {
            buffer = (float *) malloc((end_row - start_row)*ny*sizeof(float));
            MPI_Recv(buffer, (end_row - start_row) * ny, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        }
        // update rows of H for each process
        for (i = 0; i < end_row-start_row - 1; ++i)
            for (j = 0; j < ny - 1; ++j)
                buffer[(i+start_row)*ny + j] = buffer[(i+start_row)*ny + j] - 
                                                0.7f*( ex[(start_row+i)*ny + j+1] - ex[(start_row+i)*ny + j] +
                                                       ey[(start_row+i+1)*ny + j] - ey[(start_row+i)*ny + j]);
        // send result
        if (myid > 0) 
            MPI_Isend(buffer, (end_row - start_row)*ny, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, &request);
        // get result
        if (myid == 0) {
            for (int i = 1; i < numprocs; i++) {
                st_row=k*i;
                if (i==numprocs-1)
                    r_n=k+nx % numprocs;
                else 
                    r_n=k;
                MPI_Recv(&hz[st_row * ny], r_n*ny, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);
            }
        }
        else {
            free(buffer);
        }
    }
}

int main(int argc, char** argv)
{
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;
  int numprocs, myid; 

  float (*ex)[nx][ny];
  float (*ey)[nx][ny];
  float (*hz)[nx][ny];
  float (*_fict_)[tmax]; 


  MPI_Init(&argc,&argv); 
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs); 
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  if (myid==0) {
    ex = (float(*)[nx][ny])malloc((nx) * (ny) * sizeof(float));
    ey = (float(*)[nx][ny])malloc((nx) * (ny) * sizeof(float));
    hz = (float(*)[nx][ny])malloc((nx) * (ny) * sizeof(float));
    _fict_ = (float(*)[tmax])malloc((tmax) * sizeof(float));

    init_array (tmax, nx, ny,
       *ex,
       *ey,
       *hz,
       *_fict_);

    bench_timer_start();
  }

  kernel_fdtd_2d (tmax, nx, ny,
    *ex,
    *ey,
    *hz,
    *_fict_,
    myid, 
    numprocs);

  if (myid==0) {
    bench_timer_stop();
    bench_timer_print();

    free((void*)ex);
    free((void*)ey);
    free((void*)hz);
    free((void*)_fict_);
  }

  MPI_Finalize();
  return 0;
}