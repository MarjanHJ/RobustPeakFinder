// g++ -fPIC -shared -o RobustPeakFinder.so RobustPeakFinder.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
//#include <time.h>

#define INFIMUM_C			30000.0
#define MIN_STRUCT_PERCENT_C		0.5
#define PEAK_MIN_PIX			1
#define GLOBAL_THRESHOLD		0.0
#define PEAK_THRESHOLD			10.0
#define PAPR_ACCEPT_C			2.0
#define WIN_PERCENTAGE			0.8
#define	WINSIDE_MAX			8
#define	WINSIDE_MIN			4


void freeArray_d(double **a, unsigned int m) {
	int i;
	for (i = 0; i < m; ++i)
		free(a[i]);
	free(a);
}

void freeArray_i(int **a, unsigned int m) {
	int i;
	for (i = 0; i < m; ++i)
		free(a[i]);
	free(a);
}

struct sortStruct {
    double vecData;
    int indxs;
};

int partition( struct sortStruct dataVec[], int l, int r) {
   double pivot;
   int i, j;
   struct sortStruct t;

   pivot = dataVec[l].vecData;
   i = l; j = r+1;
   while(1)	{
		do ++i; while( dataVec[i].vecData <= pivot && i <= r );
		do --j; while( dataVec[j].vecData > pivot );
		if( i >= j ) break;
		t = dataVec[i];
		dataVec[i] = dataVec[j];
		dataVec[j] = t;
   }
   t = dataVec[l];
   dataVec[l] = dataVec[j];
   dataVec[j] = t;
   return j;
}

void quickSort( struct sortStruct dataVec[], int l, int r) {
   int j;
   if( l < r ) {
		j = partition( dataVec, l, r);
		quickSort( dataVec, l, j-1);
		quickSort( dataVec, j+1, r);
   }
}

bool isNotZero(int *inarray, int length){
	unsigned int i;
	for(i=0;i<length;i++)
        if(inarray[i])
			return(true);
	return(false);
}

double MSSEPeak(double *absRes, int WIN_N, double LAMBDA_C) {
	int i;
	int MSSE_FINITE_SAMPLE_BIAS;
	double estScale;
	double LAMBDA_CSq;
	double cumulative_sum, cumulative_sum_perv;
	MSSE_FINITE_SAMPLE_BIAS = floor(WIN_N/2);
	if (MSSE_FINITE_SAMPLE_BIAS < 12)
		MSSE_FINITE_SAMPLE_BIAS = 12;

	estScale = INFIMUM_C;
	LAMBDA_CSq = LAMBDA_C*LAMBDA_C;
	cumulative_sum = 0;

	for (i = 0; i < MSSE_FINITE_SAMPLE_BIAS; i++)	//finite sample bias of MSSE [RezaJMIV'06]
		cumulative_sum += absRes[i]*absRes[i];
	cumulative_sum_perv = cumulative_sum;
	for (i = MSSE_FINITE_SAMPLE_BIAS; i < WIN_N; i++) {
		if ( LAMBDA_CSq * cumulative_sum < (i-1)*absRes[i]*absRes[i])		// in (i-1), the 1 is the dimension of model
			break;
		cumulative_sum_perv = cumulative_sum;
		cumulative_sum += absRes[i]*absRes[i];
	}
	estScale = floor(1.4*sqrt(cumulative_sum_perv / (i - 2)))+1;
	return estScale;
}

////////////////// INPUTS //////////////////
//All inputs are single scalers except for a input image **Origdata
//GLOBAL_THRESHOLD    A Global Threshold
//PEAK_MIN_PIX and PEAK_MAX_PIX : number of pixels in a peak
//SNR_ACCEPT : acceptable SNR
//Origdata : 2D Matrix of imput diffraction pattern
//XPIX and YPIX : number of pixels in rows and coloumns of input image.

////////////////// OUTPUTS //////////////////
// peak_cnt is number of discovered peaks
// peaks_cheetah is a flattened matrix output size : peak_cnt by 4

////////////////// Local Peak_infor for other ML tasks ///////////////
// Peak_info = Pointer to output matrix (whose abstract is provided inthe output of the functoin):
// Rows are each peak and comloums are infor of each peak
// coloumns : 1 : MAX_NUM_PEAKS_C are the X of each pixel of a peak
//            MAX_NUM_PEAKS_C+1 : 2*MAX_NUM_PEAKS_C are the Y of each pixel of a peak
//			  2*MAX_NUM_PEAKS_C+1 : 3*MAX_NUM_PEAKS_C are the Z of each pixel of a peak
//			  3*MAX_NUM_PEAKS_C + 1 : SNR scalar value
//            3*MAX_NUM_PEAKS_C + 2 : Number of pixels in a peaksudo apt-get install atom

#ifdef __cplusplus
extern "C" {
#endif

int peakFinder(double LAMBDA_C, double SNR_ACCEPT, double *Origdata, double *originalMask, int XPIX, int YPIX, int PEAK_MAX_PIX, double *peakListCheetah) {

	int *inpData_mask;
	int *win_peak_info_x;
	int *win_peak_info_y;
	double *win_peak_info_val;
	double *win_of_peak_vec;
	int *win_of_peak_mask_vec;
	int *pix_to_visit;
	double **win_of_peak;
	int **win_of_peak_mask;

	struct sortStruct* sortVec;

	double win_estScale;
	double winModelValue;
	double sumPeakValues;
	double Peak_SNR;
	double win_Proposed_Threshold;
	double curr_pix_val;
	double Pchimg_element;
	double Pchimg_maximum;
	double Patch_Threshold;
	double Signal_Power;
	double sumWinData;
	double win_Skew;
	double tmpval;

	unsigned int WINSIDE, peakSearchCounter, not_an_extermum_flag;
	unsigned int WIN_N, WINSZ, NUM_PATCHS_ROW, NUM_PATCHS_CLM, PTCHSZ;
	unsigned int i, j, peak_pix_cnt, pixcnt, peak_cnt, win_num_pix;
	unsigned int lc_row_cnt, rcnt, Ptch_rcnt, rind, Glob_row_ind, curr_pix_x, CURX;
	unsigned int lc_clm_cnt, ccnt, Ptch_ccnt, cind, Glob_clm_ind, curr_pix_y, CURY;
	unsigned int PtchRowStart, PtchRowEnd, PtchClmStart, PtchClmEnd;
	unsigned int win_num_unmasked_pix;
	unsigned int sumNoDataPix;
	unsigned long pixelcounter, pixindex;

	Glob_clm_ind = 0;
	Glob_row_ind = 0;
	
	double **peak_info;
	unsigned int peak_info_clm = PEAK_MAX_PIX*3+2;
//clock_t start_1;
//double cpu_time_used_1=0;
//start_1 = clock();

	PTCHSZ = 40;// floor(sqrt(49 * PEAK_MAX_PIX));	//this means that size of a Peak should be less than 1% of a patch.
	NUM_PATCHS_ROW = floor(XPIX/ PTCHSZ);	//now XPIX may or may not be dividable by PTCHSZ
	NUM_PATCHS_CLM = floor(YPIX/ PTCHSZ);
	WINSIDE = WINSIDE_MAX;// floor((sqrt(9 * PEAK_MAX_PIX) - 1) / 2);
	WINSZ = 2 * WINSIDE + 1;
	WIN_N = WINSZ*WINSZ;

	win_of_peak=(double **) malloc(WINSZ*sizeof(double *));
	for(i=0;i<WINSZ;i++)
		win_of_peak[i]=(double *) malloc(WINSZ*sizeof(double));
	win_of_peak_mask=(int **) malloc(WINSZ*sizeof(int *));
	for(i=0;i<WINSZ;i++)
		win_of_peak_mask[i]=(int *) malloc(WINSZ*sizeof(int));
	inpData_mask=(int *) malloc(XPIX*YPIX*sizeof(int));
	win_of_peak_vec = (double*) malloc(WIN_N * sizeof(double));
	win_of_peak_mask_vec = (int*) malloc(WIN_N * sizeof(int));
	sortVec = (struct sortStruct*) malloc(WIN_N * sizeof(struct sortStruct));
	win_peak_info_x = (int*) malloc(WIN_N * sizeof(int));
	win_peak_info_y = (int*) malloc(WIN_N * sizeof(int));
	win_peak_info_val = (double*) malloc(WIN_N * sizeof(double));
	pix_to_visit = (int*) malloc(WIN_N * sizeof(int));
	peak_info = (double **) malloc(1*sizeof(double *));

	for (pixelcounter=0; pixelcounter<XPIX*YPIX;pixelcounter++)
		inpData_mask[pixelcounter]=originalMask[pixelcounter];

//cpu_time_used_1 += ((double) (clock() - start_1));
//cpu_time_used_1 = 0;

	//we turn the image into patches to propose peaks,
	//then, regardless of the patching, in each patch we check each proposed peak.
	peak_cnt = 0;
	for ( Ptch_rcnt = 0; Ptch_rcnt < NUM_PATCHS_ROW ; Ptch_rcnt++) {
		for ( Ptch_ccnt = 0; Ptch_ccnt < NUM_PATCHS_CLM; Ptch_ccnt++) {
//start_1 = clock();

			PtchRowStart = 0;
			PtchRowEnd = PTCHSZ;
			PtchClmStart = 0;
			PtchClmEnd = PTCHSZ;

			if (Ptch_ccnt == 0)
				PtchClmStart = 0;
			if (Ptch_ccnt == NUM_PATCHS_CLM - 1)
				PtchClmEnd = PTCHSZ + YPIX - NUM_PATCHS_CLM*PTCHSZ;
			if (Ptch_rcnt == 0)
				PtchRowStart = 0;
			if (Ptch_rcnt == NUM_PATCHS_ROW - 1)
				PtchRowEnd = PTCHSZ + XPIX - NUM_PATCHS_ROW*PTCHSZ;

			Patch_Threshold = PEAK_THRESHOLD;
			peakSearchCounter = 0;
			Pchimg_maximum = INFIMUM_C;
			while( Pchimg_maximum > Patch_Threshold) {  	//makes no difference

				Pchimg_maximum = 0;
				for (ccnt = PtchClmStart ; ccnt < PtchClmEnd ; ccnt++) {
					for (rcnt = PtchRowStart ; rcnt < PtchRowEnd ; rcnt++) {
						pixindex = (Ptch_rcnt*PTCHSZ+rcnt) + (Ptch_ccnt*PTCHSZ+ccnt)*XPIX;
						Pchimg_element = Origdata[pixindex];
						if(Pchimg_element>=Pchimg_maximum && inpData_mask[pixindex]>0) {
							Pchimg_maximum = Pchimg_element;
							Glob_row_ind = Ptch_rcnt*PTCHSZ + rcnt;   // global index of extermum
							Glob_clm_ind = Ptch_ccnt*PTCHSZ + ccnt;
						}
					}
				}
				if	(Pchimg_maximum <= PEAK_THRESHOLD)	//if a pixel was visited before or masked out for some resaon
					continue;

				peakSearchCounter++;		//count the number of local maximums

				//acquire the data around the extremum from original data.

				//now assuming a window around the pixel in orignal inp-Data and original inp-Data_mask
				//inp-Data_mask is global, copy a window of it around the pixel into win_of_peak_mask
				//later will update the win_of_peak_mask and put it back into inp-Data_mask
				i = 0;
				sumNoDataPix = 0;
				for (rcnt = 0 ; rcnt < WINSZ ; rcnt++) {
					for (ccnt = 0 ; ccnt < WINSZ ; ccnt++) {

						CURX = Glob_row_ind + rcnt - WINSIDE;
						CURY = Glob_clm_ind + ccnt - WINSIDE;

						if ((CURX < 0) || (CURX >= XPIX) || (CURY < 0) || (CURY >= YPIX)) {
							win_of_peak[rcnt][ccnt] = 0;
							win_of_peak_mask[rcnt][ccnt] = 0;
						}
						else {
							pixindex = CURX + CURY*XPIX;
							win_of_peak[rcnt][ccnt] = Origdata[pixindex];
							win_of_peak_mask[rcnt][ccnt] = inpData_mask[pixindex];
						}

						win_of_peak_vec[i] = win_of_peak[rcnt][ccnt];
						win_of_peak_mask_vec[i] = win_of_peak_mask[rcnt][ccnt];		//maybe win_of_peak_mask_vec is unnevessary
						if ((win_of_peak_mask_vec[i] == 0) || (win_of_peak_vec[i] <= GLOBAL_THRESHOLD))
							sumNoDataPix++;
						i++;
					}
				}
				pixindex = Glob_row_ind + Glob_clm_ind *XPIX;
				inpData_mask[pixindex]=0;

				not_an_extermum_flag=0;
				for (lc_row_cnt = -2 ; lc_row_cnt < 2 ; lc_row_cnt++)
					for (lc_clm_cnt = -2 ; lc_clm_cnt < 2 ; lc_clm_cnt++)
						if ((win_of_peak[WINSIDE][WINSIDE] < win_of_peak[WINSIDE+lc_row_cnt][WINSIDE+lc_clm_cnt]) && (win_of_peak_mask[WINSIDE + lc_row_cnt][WINSIDE + lc_clm_cnt]==1))
							not_an_extermum_flag=1;

				if(sumNoDataPix >= WIN_N*WIN_PERCENTAGE)
					continue;

				sumNoDataPix = 0;
				for (i=0;i<WIN_N;i++)
					if (win_of_peak_vec[i] <= GLOBAL_THRESHOLD) {
						sumNoDataPix++;
						win_of_peak_vec[i] = 2 * INFIMUM_C;
					}

				for (i = 0; i < WIN_N; i++) {
					sortVec[i].vecData  = win_of_peak_vec[i];
					sortVec[i].indxs = i;
				}
				quickSort(sortVec,0,WIN_N-1);
				winModelValue = sortVec[(int)floor(MIN_STRUCT_PERCENT_C*(double)(WIN_N - sumNoDataPix + 1))].vecData; //again could be median
				winModelValue += ((double) rand() / (RAND_MAX))/5;

				for (i = 0; i < WIN_N; i++) {
					if (sortVec[i].vecData > winModelValue)
						sortVec[i].vecData = sortVec[i].vecData - winModelValue;
					else
						sortVec[i].vecData = winModelValue - sortVec[i].vecData;
					sortVec[i].indxs = i;
				}
				quickSort(sortVec,0,WIN_N-1);
				for (i = 0; i < WIN_N; i++) {
					win_of_peak_vec[i] = sortVec[i].vecData;
				}

				win_estScale = MSSEPeak(win_of_peak_vec, WIN_N - sumNoDataPix + 1, LAMBDA_C);

				win_Proposed_Threshold = LAMBDA_C*win_estScale + winModelValue;

				if (Patch_Threshold < win_Proposed_Threshold)
					Patch_Threshold = win_Proposed_Threshold;

				if (not_an_extermum_flag>0)
					continue;

				if (win_of_peak[WINSIDE][WINSIDE] <= win_Proposed_Threshold)
					continue;


				//////////////////////////////// PAPR here:////////////////////////
				win_num_pix = 0;
				Signal_Power = 0;
				for (rcnt = 0; rcnt < WINSZ; rcnt++)
					for (ccnt = 0; ccnt < WINSZ; ccnt++) {
						if (win_of_peak[rcnt][ccnt] > GLOBAL_THRESHOLD && win_of_peak_mask[rcnt][ccnt] == 1) {
							win_num_pix++;
							Signal_Power += (win_of_peak[rcnt][ccnt] - winModelValue)*(win_of_peak[rcnt][ccnt] - winModelValue);
						}
					}
				Signal_Power = sqrt(Signal_Power / win_num_pix);

				if ((win_of_peak[WINSIDE][WINSIDE] - winModelValue) / Signal_Power <= PAPR_ACCEPT_C)
					continue;
				/////////////////////////////////////////////////////////////////

				//now begin by the extremum and mark all the adjacent
				//pixels that are above the proposed Threshold

				peak_pix_cnt = 0; //number of pixels of a peak

				//we go through adjacent pixels step by step and add them to the peak if they were above threshhold

				win_peak_info_x[peak_pix_cnt] = WINSIDE;	//this is the index of the center pixel
				win_peak_info_y[peak_pix_cnt] = WINSIDE;
				win_peak_info_val[peak_pix_cnt] = win_of_peak[WINSIDE][WINSIDE];
				sumPeakValues = win_peak_info_val[peak_pix_cnt];
				win_of_peak_mask[WINSIDE][WINSIDE] = 0;
				for(i=0;i<WIN_N;i++)
					pix_to_visit[i]=0;
				//each pixel has a flag initially off, when flag gets one, this way we know that we have to visit this new pixel later.
				pix_to_visit[peak_pix_cnt] = 1;			//here I have to visit centeral pixel

				while (isNotZero(pix_to_visit, WIN_N)) {		//check if there are any pixels left to explore
					for (pixcnt = 0 ; pixcnt <= peak_pix_cnt; pixcnt++) {	//for remaining flaged
						if (pix_to_visit[pixcnt] == 1) {
							pix_to_visit[pixcnt] = 0;
							rind = win_peak_info_x[pixcnt];
							cind = win_peak_info_y[pixcnt];
							if ( (rind==0) || (rind==WINSZ-1) || (cind==0) || (cind==WINSZ-1) )
								continue;
							for (lc_row_cnt = 0 ; lc_row_cnt < 3 ; lc_row_cnt++) {
								for (lc_clm_cnt = 0 ; lc_clm_cnt < 3 ; lc_clm_cnt++) {
									curr_pix_x = lc_row_cnt-1 + rind;
									curr_pix_y = lc_clm_cnt-1 + cind;
									if (win_of_peak_mask[curr_pix_x][curr_pix_y] == 1) {
										win_of_peak_mask[curr_pix_x][curr_pix_y] = 0;
										curr_pix_val = win_of_peak[curr_pix_x][curr_pix_y];
										if ( curr_pix_val >= win_Proposed_Threshold) {
											peak_pix_cnt++;
											win_peak_info_x[peak_pix_cnt] = curr_pix_x;
											win_peak_info_y[peak_pix_cnt] = curr_pix_y;
											win_peak_info_val[peak_pix_cnt] = curr_pix_val;
											sumPeakValues += curr_pix_val;
											pix_to_visit[peak_pix_cnt] = 1;
										}
									}
								}
							}
						}
					}
				}
				peak_pix_cnt++; // because counting starts from zero

				/////////////////////////////////////////////////////////////
				//////////// Perfomr 1 step of MeanShift ////////////////////
				sumWinData = 0;
				win_num_unmasked_pix = 0;
				for (rcnt = 0; rcnt < WINSZ; rcnt++)
					for (ccnt = 0; ccnt < WINSZ; ccnt++)
						if ((win_of_peak[rcnt][ccnt]<win_Proposed_Threshold) && (win_of_peak[rcnt][ccnt]>GLOBAL_THRESHOLD)) {
							win_num_unmasked_pix++;
							sumWinData += win_of_peak[rcnt][ccnt];
						}
				winModelValue = sumWinData/ win_num_unmasked_pix;

				win_estScale = 0;
				win_Skew = 0;
				for (rcnt = 0; rcnt < WINSZ; rcnt++)
					for (ccnt = 0; ccnt < WINSZ; ccnt++)
						if ((win_of_peak[rcnt][ccnt]<win_Proposed_Threshold) && (win_of_peak[rcnt][ccnt]>GLOBAL_THRESHOLD)) {
							tmpval = win_of_peak[rcnt][ccnt] - winModelValue;
							win_estScale += tmpval*tmpval;
							win_Skew += tmpval*tmpval*tmpval;
						}
				win_estScale = sqrt(win_estScale/(win_num_unmasked_pix-1));

				Peak_SNR = (win_of_peak[WINSIDE][WINSIDE] - winModelValue) / win_estScale;

				if ( (peak_pix_cnt >= PEAK_MIN_PIX) && (peak_pix_cnt <= PEAK_MAX_PIX) && (Peak_SNR > SNR_ACCEPT) ) {

					if (peak_cnt)
						peak_info = (double **) realloc(peak_info, (peak_cnt+1)*sizeof(double *));
					peak_info[peak_cnt]=(double *) malloc( peak_info_clm*sizeof(double));
					for(j=0;j<peak_info_clm;j++)
						peak_info[peak_cnt][j]=0;

					//Our own Peaklist
					double mass_x=0;
					double mass_y=0;
					double mass_t=0;
					for(i=0;i<peak_pix_cnt;i++) {
						mass_x += (win_peak_info_x[i] - WINSIDE + Glob_row_ind + 1)*(win_peak_info_val[i]);
						mass_y += (win_peak_info_y[i] - WINSIDE + Glob_clm_ind + 1)*(win_peak_info_val[i]);
						mass_t += win_peak_info_val[i];
						peak_info[peak_cnt][i] = win_peak_info_x[i] - WINSIDE + Glob_row_ind + 1; // global x of founded pixeles of peak
						peak_info[peak_cnt][i+PEAK_MAX_PIX] = win_peak_info_y[i] - WINSIDE + Glob_clm_ind  + 1; // y of founded pixeles of peak
						peak_info[peak_cnt][i+2*PEAK_MAX_PIX] = win_peak_info_val[i]; // value
					}
					//peak_info[peak_cnt][3*PEAK_MAX_PIX-2] = winModelValue;
					//peak_info[peak_cnt][3*PEAK_MAX_PIX-1] = win_estScale;
					peak_info[peak_cnt][3*PEAK_MAX_PIX] = Peak_SNR; // SNR
					peak_info[peak_cnt][3*PEAK_MAX_PIX+1] = peak_pix_cnt; // number of pixels

					//Complying with Cheetah's output
					peakListCheetah[4*peak_cnt+0] = mass_x/mass_t - 0.5;
					peakListCheetah[4*peak_cnt+1] = mass_y/mass_t - 0.5;
					peakListCheetah[4*peak_cnt+2] = mass_t;
					peakListCheetah[4*peak_cnt+3] = peak_pix_cnt;

					peak_cnt++;

				}

				for (rcnt = 0 ; rcnt < WINSZ ; rcnt++) {
					for (ccnt = 0 ; ccnt < WINSZ ; ccnt++) {
						CURX = Glob_row_ind + rcnt - WINSIDE;
						CURY = Glob_clm_ind + ccnt - WINSIDE;
						if ((CURX >= 0) && (CURX < XPIX) && (CURY >= 0) && (CURY < YPIX)) {
							pixindex = CURX + CURY*XPIX;
							inpData_mask[pixindex] = win_of_peak_mask[rcnt][ccnt];
						}
					}
				}

			}	//end of while(peaks)
//cpu_time_used_1 += ((double) (clock() - start_1));

		} //end of for pathes_y
	} //end of for pathes_x
//cpu_time_used_1 = cpu_time_used_1/CLOCKS_PER_SEC;

freeArray_d(win_of_peak,WINSZ);
freeArray_i(win_of_peak_mask,WINSZ);
freeArray_d(peak_info, peak_cnt);

free(inpData_mask);
free(win_of_peak_vec);
free(win_of_peak_mask_vec);
free(sortVec);
free(win_peak_info_x);
free(win_peak_info_y);
free(win_peak_info_val);
free(pix_to_visit);

return(peak_cnt);
}


#ifdef __cplusplus
}
#endif
