/*
 * hw8.h
 *
 *  Created on: 10 Jun 2020
 *      Author: paz
 */
#pragma once
#ifndef HW8_H_
#define HW8_H_
#define LINE "\n================================================================================================================================================================================"
#define MSG1 "\n================================================================================ Histogram Array ==============================================================================="
#define MSG2 "\n=========================================================================== Histogram Vertical Graph ==========================================================================="

#define  MASTER 0
#define SLAVE 1
#define MAX_PROC 2
#define RANGE_SIZE 256

void initArrays(char *filename , int** initial_array, int** received_array , int* size);
int* OpenMPTask(int* src_arr , int size);
int* CUDATask(int* arr, int size , int arr_size);
void OpenMPFinalMergeTask(int* dest_array, int* src_array, int size);
void PrintHistogram(int* arr);
void myIntArrCalloc(int** arr , int size);
void printRes(int* res , int size);
void printHeadline(int msg_num);
#endif /* HW8_H_ */
