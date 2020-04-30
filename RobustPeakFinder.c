// g++ -fPIC -shared -o RobustPeakFinder.so RobustPeakFinder.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
//#include <time.h>
#include "./../RobustGausFitLib/RobustGausFitLib.c"

void freeArray_f(float **a, unsigned int m) {
	unsigned int i;
	for (i = 0; i < m; ++i)
		free(a[i]);
	free(a);
}

void freeArray_ub(unsigned char **a, unsigned int m) {
	unsigned int i;
	for (i = 0; i < m; ++i)
		free(a[i]);
	free(a);
}

bool isNotZero(int *inarray, int length){
	unsigned int i;
	for(i=0;i<length;i++)
        if(inarray[i])
			return(true);
	return(false);
}

#ifdef __cplusplus
extern "C" {
#endif

int peakFinder(	float *inData, unsigned char *inMask,
				float *SNRFactor, float *minPeakValMap, float *maxBackMeanMap, 
				float *peakList, int MAXIMUM_NUMBER_OF_PEAKS,
				float bckSNR, float pixPAPR,
				int XPIX, int YPIX, int PTCHSZ,	
				int PEAK_MIN_PIX, int PEAK_MAX_PIX) {

	unsigned char *inpData_mask;
	int *win_peak_info_x;
	int *win_peak_info_y;
	float *win_peak_info_val;
	float *win_of_peak_vec;
	unsigned char *win_of_peak_mask_vec;
	int *pix_to_visit;
	float **win_of_peak;
	unsigned char **win_of_peak_mask;

	float win_estScale;
	float winModelValue;
	float sumPeakValues;
	float Peak_SNR;
	float win_Proposed_Threshold;
	float curr_pix_val;
	float pixValue;
	float Pchimg_maximum;
	float Patch_Threshold;
	float Signal_Power;
	float modelParams[2];
	float mass_x;
	float mass_y;
	float mass_t;

	int lc_row_cnt, lc_clm_cnt;
	unsigned int WINSIDE, not_an_extermum_flag;
	unsigned int WIN_N, WINSZ, NUM_PATCHS_ROW, NUM_PATCHS_CLM;
	unsigned int i, peak_pix_cnt, pixcnt, peak_cnt, win_num_pix;
	unsigned int rcnt, Ptch_rcnt, rind, Glob_row_ind, curr_pix_x, CURX;
	unsigned int ccnt, Ptch_ccnt, cind, Glob_clm_ind, curr_pix_y, CURY;
	unsigned int PtchRowStart, PtchRowEnd, PtchClmStart, PtchClmEnd;
	unsigned int sumNoDataPix;
	unsigned long pixelcounter, pixIndex;
	unsigned char dist2Max;

//clock_t start_1;
//float cpu_time_used_1=0;
//start_1 = clock();

	//PTCHSZ = 16;// floor(sqrt(49 * PEAK_MAX_PIX));	//this means that size of a Peak should be less than 1% of a patch.
		// we chose 64 for AGIPD detector where the ASIC size is 64
	NUM_PATCHS_ROW = floor(XPIX/ PTCHSZ);	//now XPIX may or may not be dividable by PTCHSZ
	NUM_PATCHS_CLM = floor(YPIX/ PTCHSZ);
	WINSIDE = (int) floor(PTCHSZ/2)+1;
	WINSZ = 2 * WINSIDE + 1;
	WIN_N = WINSZ*WINSZ;

	win_of_peak=(float **) malloc(WINSZ*sizeof(float *));
	for(i=0;i<WINSZ;i++)
		win_of_peak[i]=(float *) malloc(WINSZ*sizeof(float));
	win_of_peak_mask=(unsigned char **) malloc(WINSZ*sizeof(unsigned char *));
	for(i=0;i<WINSZ;i++)
		win_of_peak_mask[i]=(unsigned char *) malloc(WINSZ*sizeof(unsigned char));
	inpData_mask=(unsigned char *) malloc(XPIX*YPIX*sizeof(unsigned char));
	win_of_peak_vec = (float*) malloc(WIN_N * sizeof(float));
	win_of_peak_mask_vec = (unsigned char*) malloc(WIN_N * sizeof(unsigned char));
	win_peak_info_x = (int*) malloc(WIN_N * sizeof(int));
	win_peak_info_y = (int*) malloc(WIN_N * sizeof(int));
	win_peak_info_val = (float*) malloc(WIN_N * sizeof(float));
	pix_to_visit = (int*) malloc(WIN_N * sizeof(int));

	for (pixelcounter=0; pixelcounter<XPIX*YPIX;pixelcounter++)
		inpData_mask[pixelcounter]=inMask[pixelcounter];

//cpu_time_used_1 += ((float) (clock() - start_1));
//cpu_time_used_1 = 0;

	//we turn the image into patches to propose peaks,
	//then, regardless of the patching, in each patch we check each proposed peak.
	Glob_row_ind = 0;
	Glob_clm_ind = 0;
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

			Patch_Threshold = 0;
			Pchimg_maximum = Patch_Threshold + 1;
			while( Patch_Threshold < Pchimg_maximum ) {
			
				Pchimg_maximum = 0;
				for (ccnt = PtchClmStart ; ccnt < PtchClmEnd ; ccnt++) {
					for (rcnt = PtchRowStart ; rcnt < PtchRowEnd ; rcnt++) {
						pixIndex = (Ptch_rcnt*PTCHSZ+rcnt) + (Ptch_ccnt*PTCHSZ+ccnt)*XPIX;
						pixValue = inData[pixIndex];
						if( (pixValue>=Pchimg_maximum) && (inpData_mask[pixIndex]>0) ) {
							Pchimg_maximum = pixValue;
							Glob_row_ind = Ptch_rcnt*PTCHSZ + rcnt;   // global index of extermum
							Glob_clm_ind = Ptch_ccnt*PTCHSZ + ccnt;
						}
					}
				}
				if(Pchimg_maximum <= 1)
					break;
				pixIndex = Glob_row_ind + Glob_clm_ind *XPIX;
				
				//if the patch maximum is masked or too small
				if ( (Pchimg_maximum <= Patch_Threshold) ||
					 (Pchimg_maximum <= minPeakValMap[pixIndex]) )
						break;

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
							win_of_peak[rcnt][ccnt] = inData[CURX + CURY*XPIX];
							win_of_peak_mask[rcnt][ccnt] = inpData_mask[CURX + CURY*XPIX];
						}

						win_of_peak_vec[i] = win_of_peak[rcnt][ccnt];
						win_of_peak_mask_vec[i] = win_of_peak_mask[rcnt][ccnt];		//maybe win_of_peak_mask_vec is unnecessary
						if (win_of_peak_mask_vec[i] == 0)
							sumNoDataPix++;
						i++;
					}
				}
				inpData_mask[pixIndex]=0;
				// 78: we would love to capture the background of a bragg peak that 
				// can spread at lest to one pixel close 
				// to the local maximum. Lets say all 9 pixels are occupied. 
				// Then at least thress pixels away must be present for background estimation. So that is:
				// (9*4)-2 + (7*4)-2 + (5*4)-2 = 78 data points
				if(WIN_N - sumNoDataPix < 78)	
					continue;

				not_an_extermum_flag=0;
				for (lc_row_cnt = -2 ; lc_row_cnt < 2 ; lc_row_cnt++)
					for (lc_clm_cnt = -2 ; lc_clm_cnt < 2 ; lc_clm_cnt++) 
						if (win_of_peak[WINSIDE][WINSIDE] < win_of_peak[WINSIDE+lc_row_cnt][WINSIDE+lc_clm_cnt])
							not_an_extermum_flag=1;
				if (not_an_extermum_flag>0)
					continue;

				
				RobustSingleGaussianVec(win_of_peak_vec, modelParams, WIN_N, 0.5, 0.4, bckSNR);
				winModelValue = modelParams[0];
				win_estScale = modelParams[1];
				
				win_Proposed_Threshold = bckSNR*win_estScale + winModelValue;
				
				if (Patch_Threshold < win_Proposed_Threshold)
					Patch_Threshold = win_Proposed_Threshold;

				if (win_of_peak[WINSIDE][WINSIDE] <= win_Proposed_Threshold)
					continue;

				if (winModelValue > maxBackMeanMap[pixIndex])
					continue;

				//////////////////////////////// PAPR here:////////////////////////
				win_num_pix = 0;
				Signal_Power = 0;
				for (rcnt = 0; rcnt < WINSZ; rcnt++)
					for (ccnt = 0; ccnt < WINSZ; ccnt++) {
						if (win_of_peak[rcnt][ccnt] > (winModelValue - bckSNR*win_estScale) && 
							win_of_peak_mask[rcnt][ccnt] == 1) {
							win_num_pix++;
							Signal_Power += (win_of_peak[rcnt][ccnt] - winModelValue)*(win_of_peak[rcnt][ccnt] - winModelValue);
						}
					}
				Signal_Power = sqrt(Signal_Power / win_num_pix);
				if ( ((win_of_peak[WINSIDE][WINSIDE] - winModelValue) / Signal_Power) <= pixPAPR)
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
				dist2Max = 0;
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
									dist2Max = (curr_pix_x-WINSIDE)*(curr_pix_x-WINSIDE)+(curr_pix_y-WINSIDE)*(curr_pix_y-WINSIDE);
									if (win_of_peak_mask[curr_pix_x][curr_pix_y] == 1) {
										win_of_peak_mask[curr_pix_x][curr_pix_y] = 0;
										curr_pix_val = win_of_peak[curr_pix_x][curr_pix_y];
										if ( curr_pix_val >= win_of_peak[WINSIDE][WINSIDE]*exp(-dist2Max/2)) {
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
				}
				peak_pix_cnt++; // because counting starts from zero
				
				Peak_SNR = (win_of_peak[WINSIDE][WINSIDE] - winModelValue) / (bckSNR * win_estScale);
				Peak_SNR = SNRFactor[pixIndex]*Peak_SNR;	//yes! you need probability map for bad pixel mask
				// This can be learned over background runs.

				if ( (peak_pix_cnt >= PEAK_MIN_PIX) && (peak_pix_cnt <= PEAK_MAX_PIX) && 
					 (Peak_SNR > 1) && (peak_cnt<MAXIMUM_NUMBER_OF_PEAKS)) {
					mass_x = 0;
					mass_y = 0;
					mass_t = 0;
					for(i=0;i<peak_pix_cnt;i++) {
						mass_x += (win_peak_info_x[i] - WINSIDE + Glob_row_ind + 1)*(win_peak_info_val[i]);
						mass_y += (win_peak_info_y[i] - WINSIDE + Glob_clm_ind + 1)*(win_peak_info_val[i]);
						mass_t += win_peak_info_val[i];
					}
					//Complying with Cheetah's output
					peakList[6*peak_cnt+0] = mass_x/mass_t - 1;
					peakList[6*peak_cnt+1] = mass_y/mass_t - 1;
					peakList[6*peak_cnt+2] = mass_t;
					peakList[6*peak_cnt+3] = peak_pix_cnt;
					peakList[6*peak_cnt+4] = win_of_peak[WINSIDE][WINSIDE];
					peakList[6*peak_cnt+5] = Peak_SNR;
					peak_cnt++;
				}

				for (rcnt = 0 ; rcnt < WINSZ ; rcnt++) {
					for (ccnt = 0 ; ccnt < WINSZ ; ccnt++) {
						CURX = Glob_row_ind + rcnt - WINSIDE;
						CURY = Glob_clm_ind + ccnt - WINSIDE;
						if ((CURX >= 0) && (CURX < XPIX) && (CURY >= 0) && (CURY < YPIX)) {
							pixIndex = CURX + CURY*XPIX;
							inpData_mask[pixIndex] = win_of_peak_mask[rcnt][ccnt];
						}
					}
				}
			}	//end of while(peaks)
//cpu_time_used_1 += ((float) (clock() - start_1));

		} //end of for pathes_y
	} //end of for pathes_x
//cpu_time_used_1 = cpu_time_used_1/CLOCKS_PER_SEC;

freeArray_f(win_of_peak, WINSZ);
freeArray_ub(win_of_peak_mask, WINSZ);

free(inpData_mask);
free(win_of_peak_vec);
free(win_of_peak_mask_vec);

free(win_peak_info_x);
free(win_peak_info_y);
free(win_peak_info_val);
free(pix_to_visit);

return(peak_cnt);
}


#ifdef __cplusplus
}
#endif
