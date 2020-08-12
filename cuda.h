/*
 * cuda.h
 *
 *  Created on: 10 Jun 2020
 *      Author: paz
 */
#pragma once

#ifndef CUDA_H_
#define CUDA_H_
#include <math.h>
#include "hw8.h"
#define THREADS 1000
int* calculateHistogramm(int *image, unsigned int size , int arr_size);
#endif /* CUDA_H_ */
