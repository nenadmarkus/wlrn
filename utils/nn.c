#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include <string.h>

/*
	portable time function
*/

#ifdef __GNUC__
#include <time.h>
float getticks()
{
	struct timespec ts;

	if(clock_gettime(CLOCK_MONOTONIC, &ts) < 0)
	{
		printf("# clock_gettime error\n");

		return -1.0f;
	}

	return ts.tv_sec + 1e-9f*ts.tv_nsec;
}
#else
#include <windows.h>
float getticks()
{
	static double freq = -1.0;
	LARGE_INTEGER lint;

	if(freq < 0.0)
	{
		if(!QueryPerformanceFrequency(&lint))
			return -1.0f;

		freq = lint.QuadPart;
	}

	if(!QueryPerformanceCounter(&lint))
		return -1.0f;

	return (float)( lint.QuadPart/freq );
}
#endif

/*
	
*/

int get_score_h(void* bag1, void* bag2)
{
	int i, n;
	float sim, sv1, sv2;

	float* v1;
	float* v2;

	//
	if(*(int*)bag1 != *(int*)bag2)
		return 0;

	n = *(int*)bag1;

	//
	v1 = (float*)(1+(int*)bag1);
	v2 = (float*)(1+(int*)bag2);

	sim = 0.0f;
	sv1 = 0.0f;
	sv2 = 0.0f;

	for(i=0; i<n; ++i)
	{
		//
		sim = sim + v1[i]*v2[i];

		//
		sv1 = sv1 + v1[i]*v1[i];
		sv2 = sv2 + v2[i]*v2[i];
	}

	//
	int score;

	if(sv1<=0.0f || sv2<=0.0f)
		score = 0;
	else
		score = (int)(1024.0f*(0.5f + 0.5f*sim/sqrtf(sv1)/sqrtf(sv2)));

	return score;

	/*
	sim = 0.0f;

	for(i=0; i<n; ++i)
		if(v1[i] < v2[i])
			sim = sim + v1[i];
		else
			sim = sim + v2[i];

	return (int)(1024*sim);
	*/
}

/*
	
*/

float edist(float v1[], float v2[], int ndims)
{
	float accum;
	int i;

	//
	accum = 0.0f;

	for(i=0; i<ndims; ++i)
		accum += (v1[i]-v2[i])*(v1[i]-v2[i]);

	//
	return sqrt(accum);
}

int get_score_f(void* bag1, void* bag2, float t)
{
	int i, j, n, n1, n2, ndims;

	float* s1;
	float* s2;

	float d;

	//
	ndims = *(int*)bag1;

	if(ndims != *(int*)bag2)
		return 0;

	//
	n = 0;

	if(*(1+(int*)bag1) < *(1+(int*)bag2))
	{
		n1 = *(1+(int*)bag1);
		s1 = (float*)( 2+(int*)bag1 );

		n2 = *(1+(int*)bag2);
		s2 = (float*)( 2+(int*)bag2 );
	}
	else
	{
		n1 = *(1+(int*)bag2);
		s1 = (float*)( 2+(int*)bag2 );

		n2 = *(1+(int*)bag1);
		s2 = (float*)( 2+(int*)bag1 );
	}

	for(i=0; i<n1; ++i)
		for(j=0; j<n2; ++j)
		{
			//
			d = edist(&s1[i*ndims], &s2[j*ndims], ndims);

			//
			if(d < t)
			{
				++n;
				break;
			}
		}

	//
	//return n;
	return 100*n/n1;
}

/*
	
*/

int hamm(uint8_t s1[], uint8_t s2[], int nbytes)
{
	int i, h;

	static int popcntlut[256] =
	{
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
	};

	//
	h = 0;

	for(i=0; i<nbytes; ++i)
		h = h + popcntlut[s1[i]^s2[i]];

	//
	return h;
}

int get_score_b(void* bag1, void* bag2, int t)
{
	int i, j, k, h, n, nbytes;

	uint8_t* s1;
	uint8_t* s2;

	//
	nbytes = *(int*)bag1;

	if(nbytes != *(int*)bag2)
		return 0;

	//
	n = 0;

	int n1, n2;

	if(*(1+(int*)bag1) < *(1+(int*)bag2))
	{
		n1 = *(1+(int*)bag1);
		s1 = (uint8_t*)( 2+(int*)bag1 );

		n2 = *(1+(int*)bag2);
		s2 = (uint8_t*)( 2+(int*)bag2 );
	}
	else
	{
		n1 = *(1+(int*)bag2);
		s1 = (uint8_t*)( 2+(int*)bag2 );

		n2 = *(1+(int*)bag1);
		s2 = (uint8_t*)( 2+(int*)bag1 );
	}

	for(i=0; i<n1; ++i)
		for(j=0; j<n2; ++j)
		{
			//
			h = hamm(&s1[i*nbytes], &s2[j*nbytes], nbytes);

			//
			if(h < t)
			{
				++n;
				break;
			}
		}

	//
	return n;
}

/*
	
*/

char magic[4] = "";
int ndims=0, nbags=0;

char labels[16384][32];
void* bags[16384];

int load_bags(char* path)
{
	char buffer[1024];
	FILE* file;

	//
	sprintf(buffer, "%s/list", path);

	file = fopen(buffer, "r");

	if(!file)
	{
		printf("* cannot open '%s'\n", buffer);
		return 0;
	}

	//
	nbags = 0;

	while(1 == fscanf(file, "%s", &labels[nbags][0]))
	{
		FILE* tmpfile;
		char tmpbuffer[1024];

		//
		sprintf(tmpbuffer, "%s/%s", path, &labels[nbags][0]);

		tmpfile = fopen(tmpbuffer, "rb");

		if(tmpfile)
		{
			int n;

			//
			fread(magic, 1, 4, tmpfile);

			if(magic[0]=='f')
			{
				//
				fread(&ndims, 1, sizeof(int), tmpfile);
				fread(&n, 1, sizeof(int), tmpfile);

				//
				bags[nbags] = malloc(2*sizeof(int)+ndims*n*sizeof(float));

				//
				*(int*)bags[nbags] = ndims;
				*(1+(int*)bags[nbags]) = n;

				//
				fread(2+(int*)bags[nbags], sizeof(float), n*ndims, tmpfile);
			}
			else if(magic[0]=='b')
			{
				//
				fread(&ndims, 1, sizeof(int), tmpfile);
				fread(&n, 1, sizeof(int), tmpfile);

				//
				bags[nbags] = malloc(2*sizeof(int)+ndims*n*sizeof(uint8_t));

				//
				*(int*)bags[nbags] = ndims;
				*(1+(int*)bags[nbags]) = n;

				//
				fread(2+(int*)bags[nbags], sizeof(uint8_t), n*ndims, tmpfile);
			}
			else
			{
				//
				fread(&ndims, 1, sizeof(int), tmpfile);

				//
				bags[nbags] = malloc(1*sizeof(int)+ndims*sizeof(float));

				//
				*(int*)bags[nbags] = ndims;

				//
				fread(1+(int*)bags[nbags], sizeof(float), ndims, tmpfile);
			}

			//
			fclose(tmpfile);

			//
			char* str = &labels[nbags][0];
			while(*str!='.')
				++str;
			*str = '\0';

			//
			++nbags;
		}
	}

	fclose(file);

	//
	return 1;
}

/*
	
*/

int S[16384][16384];

void compute_similarity_matrix(float thr)
{
	int i;

	#pragma omp parallel for
	for(i=0; i<nbags; ++i)
	{
		int j;

		for(j=0; j<nbags; ++j)
			/*if(i==j)
				S[i][j] = 0;
			else
			*/
			{
				int score;

				//
				if(magic[0] == 'b')
					score = get_score_b(bags[i], bags[j], (int)thr);
				else if(magic[0] == 'f')
					score = get_score_f(bags[i], bags[j], thr);
				else
					score = get_score_h(bags[i], bags[j]);

				//
				S[i][j] = score;
			}
	}

	// print similarity matrix
	/*
	printf("\n-------------------\n");
	for(i=0; i<nbags; ++i)
	{
		int j;

		//
		for(j=0; j<nbags; ++j)
			printf("%d\t", S[i][j]);

		//
		printf("\n");
	}
	printf("-------------------\n");
	//*/
}

int load_similarity_matrix(const char path[])
{
	int i, j;
	FILE* file;

	//
	file = fopen(path, "r");

	if(!file)
		return 0;

	//
	magic[0] = 'm';
	magic[1] = 't';
	magic[2] = 'r';
	magic[3] = 'x';

	//
	fscanf(file, "%d", &nbags);

	for(i=0; i<nbags; ++i)
	{
		//
		fscanf(file, "%s", &labels[i][0]);

		//
		char* str = &labels[i][0];
		while(*str!='\0' && *str!='.')
			++str;
		*str = '\0';

		//
		//printf("%s\n", labels[i]);
	}

	for(i=0; i<nbags; ++i)
		for(j=0; j<nbags; ++j)
			fscanf(file, "%d", &S[i][j]);

	//
	fclose(file);

	//
	return 1;
}

/*
	
*/

void nn(float* onenn, float* t1, float* t2, float* avgp, float* avgn)
{
	int i, ncorrect[3], nretrieved[3], nrelevant[3];

	int np, nn;

	//
	ncorrect[0] = 0;
	ncorrect[1] = 0;
	ncorrect[2] = 0;

	nretrieved[0] = 0;
	nretrieved[1] = 0;
	nretrieved[2] = 0;

	nrelevant[0] = 0;
	nrelevant[1] = 0;
	nrelevant[2] = 0;

	nn = np = 0;

	*avgp = 0.0f;
	*avgn = 0.0f;

	//
	//#pragma omp parallel for
	for(i=0; i<nbags; ++i)
	{
		int n, nc, j, k, l, topinds[128], topscores[128];

		// get the size of the class to which bag[i] belongs
		nc = 0;

		for(j=0; j<nbags; ++j)
			if(0==strcmp(&labels[i][0], &labels[j][0]))
				++nc;

		//
		n = 2*(nc-1);

		for(k=0; k<n; ++k)
			topscores[k] = -1;

		for(j=0; j<nbags; ++j)
			if(i!=j)
			{
				int score;

				//
				score = S[i][j];

				//
				topscores[n] = -1;
				k=0;
				while(score <= topscores[k])
					++k;

				for(l=n-1; l>k; --l)
				{
					topscores[l] = topscores[l-1];
					topinds[l] = topinds[l-1];
				}
				topscores[k] = score;
				topinds[k] = j;

				//
				//#pragma omp critical
				{
					if(0==strcmp(&labels[i][0], &labels[j][0]))
					{
						++np;
						*avgp = *avgp + score;
					}
					else
					{
						++nn;
						*avgn = *avgn + score;
					}
				}
			}

		//
		//#pragma omp critical
		{
			// nn
			if(0==strcmp(&labels[i][0], &labels[topinds[0]][0]))
				++ncorrect[0];

			nretrieved[0] += 1;
			nrelevant[0] += nc-1;

			// 1st tier
			for(k=0; k<nc-1; ++k)
				if(0==strcmp(&labels[i][0], &labels[topinds[k]][0]))
					++ncorrect[1];

			nretrieved[1] += nc-1;
			nrelevant[1] += nc-1;

			// 2nd tier
			for(k=0; k<2*(nc-1); ++k)
				if(0==strcmp(&labels[i][0], &labels[topinds[k]][0]))
					++ncorrect[2];

			nretrieved[2] += 2*(nc-1);
			nrelevant[2] += nc-1;
		}
	}

	//
	*onenn = ncorrect[0]/(float)nretrieved[0];

	*t1 = ncorrect[1]/(float)nrelevant[1];
	*t2 = ncorrect[2]/(float)nrelevant[2];

	*avgp = *avgp/np;
	*avgn = *avgn/nn;
}

/*
	
*/

int main(int argc, char** argv)
{
	float t, thr, t1, t2, onenn, avgp, avgn;

	if(argc==2)
	{
		//
		t = getticks();

		if(!load_similarity_matrix(argv[1]))
			return 0;

		//
		nn(&onenn, &t1, &t2, &avgp, &avgn);
		printf("(t: %d [s]):   nn = %f   T1 = %f   T2 = %f\n", (int)(getticks()-t), onenn, t1, t2);

		//
		return 0;
	}
	else if(argc!=3)
	{
		printf("* source folder\n");
		printf("* threshold\n");
		return 0;
	}

	//
	if(!load_bags(argv[1]))
		return 0;

	sscanf(argv[2], "%f", &thr);

	//
	if(nbags == 2)
	{
		//
		printf("%d\n", get_score_b(bags[0], bags[1], thr) );

		//
		return 0;
	}

	//
	if(magic[0] == 'b')
		printf("* %c%c%c%c%d@%d ", magic[0], magic[1], magic[2], magic[3], 8*ndims, (int)thr);
	else
		printf("* %c%c%c%c%d@%f ", magic[0], magic[1], magic[2], magic[3], ndims, thr);

	//
	t = getticks();
	compute_similarity_matrix(thr);
	nn(&onenn, &t1, &t2, &avgp, &avgn);
	printf("(t: %d [s]):   NN = %f   FT = %f   ST = %f\n", (int)(getticks()-t), onenn, t1, t2);

	//
	return 0;
}
