
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#if __AARCH64EL__ == 1
typedef long double Dll;
#else
typedef struct {Ull u[2];} Dll;
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include "tensor.h"
#include "cnnet.h"
#include "monitor.h"

#ifdef CBLAS_GEMM
#include "cblas.h"
#endif

#include "./emax7.h"
#include "./emax7lib.c"

void *memcpy();
int soft32(Uint, float, float, float, float *);
int hard32(Uint, float, float, float, float *, Uint);
int soft64(Uint, float, float, float, float *);
int hard64(Uint, float, float, float, float *);

extern struct c c[2][CNN_DEPTH_MAX];
extern struct f f[2][FC_DEPTH_MAX];
extern int      CNN_DEPTH;/* default 1   */
extern int      FC_DEPTH; /* default 1   */

Uchar   *membase;
//Uchar   *membaseX;
int     memsize;
int     memalign;

Uint    *i_inp[EMAX_NLANE]; /* for CNN on ZYNQ_PL */
Uint    *i_ker[EMAX_NLANE]; /* for CNN on ZYNQ_PL */
Uint    *i_out[EMAX_NLANE]; /* for CNN on ZYNQ_PL */
int     i_inp_max_size;
int     i_ker_max_size;
int     i_out_max_size;
//Uint  *i_inpX[EMAX_NLANE]; /* for CNN on ZYNQ_PL */
//Uint  *i_kerX[EMAX_NLANE]; /* for CNN on ZYNQ_PL */
//Uint  *i_outX[EMAX_NLANE]; /* for CNN on ZYNQ_PL */

Uint    *i_m0A[EMAX_NLANE]; /* for sgemm00 on ZYNQ_PL */
Uint    *i_m0B[EMAX_NLANE]; /* for sgemm00 on ZYNQ_PL */
Uint    *i_m0C[EMAX_NLANE]; /* for sgemm00 on ZYNQ_PL */
int     i_m0A_max_size;
int     i_m0B_max_size;
int     i_m0C_max_size;

#define ERRTH  (5.0E-2)
#define udiff(a,b) (((a)-(b)>=0.0?(a)-(b):(b)-(a))/((a)==0.0?1:(a)))
#define setmax(max, new) { if (max < (new)) max = (new); }

void init_xmax(int batch_size, struct c *c, struct f *f)
{
  int l;

  for (l=0; l<CNN_DEPTH; l++) {
    setmax(i_inp_max_size, batch_size * c[l].ichan * (c[l].isize+c[l].ksize-1) * (c[l].isize+c[l].ksize-1));
    setmax(i_ker_max_size, c[l].ichan * ((c[l].ochan+3)&~3) * c[l].ksize * c[l].ksize);
    setmax(i_out_max_size, batch_size * ((c[l].ochan+3)&~3) * c[l].osize * c[l].osize);
  }
  /* sgemm00(!transA&&!transB)¤Ïforward¤Îfc¤Î¤ß³ÎÊÝ¤ÇOK */
  for (l=0; l<FC_DEPTH; l++) {
    setmax(i_m0A_max_size, batch_size * ((l==0)?c[CNN_DEPTH-1].ochan * c[CNN_DEPTH-1].osize * c[CNN_DEPTH-1].osize:f[FC_DEPTH_MAX-FC_DEPTH+l-1].osize));
    setmax(i_m0B_max_size,              ((l==0)?c[CNN_DEPTH-1].ochan * c[CNN_DEPTH-1].osize * c[CNN_DEPTH-1].osize:f[FC_DEPTH_MAX-FC_DEPTH+l-1].osize) * f[FC_DEPTH_MAX-FC_DEPTH+l].osize);
    setmax(i_m0C_max_size, batch_size *                                                                                                                 (f[FC_DEPTH_MAX-FC_DEPTH+l].osize+3)&~3);
  }
  setmax(memsize, (i_inp_max_size+i_ker_max_size+i_out_max_size)*sizeof(int));
  setmax(memsize, (i_m0A_max_size+i_m0B_max_size+i_m0C_max_size)*sizeof(int));
  memalign = 32;

#if defined(EMAX7)
#if defined(ARMZYNQ)
  if ((NLANE = emax7_open(8)) == NULL) /* EMAX7 MACRO_PIPELIING (tsim-zynq,tsim-zynq.emax7*) */
    exit(1);
#else
  NLANE = 8; /* EMAX7 MACRO_PIPELIING (tsim-bsd.emax7nc,tsim-cent.emax7nc) */
#endif
#endif
#if defined(EMAX7) && defined(ARMZYNQ)
  membase = emax_info[0].ddr_mmap;
//posix_memalign(&membaseX, memalign, memsize*NLANE);
  /*{int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}*/
#elif __linux__ == 1
  posix_memalign(&membase, memalign, memsize*NLANE);
#else
  membase = (void*)malloc(memsize*NLANE+memalign);
  if ((Ull)membase & (Ull)(memalign-1))
    membase = (void*)(((Ull)membase & ~(Ull)(memalign-1))+memalign);
#endif

  printf("membase: %08.8x\n", (Uint)membase);

  for (l=0; l<NLANE; l++) {
    i_inp[l] = (Uint*)membase+memsize/sizeof(int)*l;
    i_ker[l] = (Uint*)i_inp[l] + i_inp_max_size;
    i_out[l] = (Uint*)i_ker[l] + i_ker_max_size;
  //i_inpX[l] = (Uint*)membaseX+memsize/sizeof(int)*l;
  //i_kerX[l] = (Uint*)i_inpX[l] + i_inp_max_size;
  //i_outX[l] = (Uint*)i_kerX[l] + i_ker_max_size;
    printf("i_inp[%d] : %08.8x-%08.8x\n", l, (Uint)i_inp[l], (Uint)i_inp[l]+i_inp_max_size*sizeof(int)-1);
    printf("i_ker[%d] : %08.8x-%08.8x\n", l, (Uint)i_ker[l], (Uint)i_ker[l]+i_ker_max_size*sizeof(int)-1);
    printf("i_out[%d] : %08.8x-%08.8x\n", l, (Uint)i_out[l], (Uint)i_out[l]+i_out_max_size*sizeof(int)-1);
    i_m0A[l] = (Uint*)membase+memsize/sizeof(int)*l;
    i_m0B[l] = (Uint*)i_m0A[l] + i_m0A_max_size;
    i_m0C[l] = (Uint*)i_m0B[l] + i_m0B_max_size;
    printf("i_m0A[%d] : %08.8x-%08.8x\n", l, (Uint)i_m0A[l], (Uint)i_m0A[l]+i_m0A_max_size*sizeof(int)-1);
    printf("i_m0B[%d] : %08.8x-%08.8x\n", l, (Uint)i_m0B[l], (Uint)i_m0B[l]+i_m0B_max_size*sizeof(int)-1);
    printf("i_m0C[%d] : %08.8x-%08.8x\n", l, (Uint)i_m0C[l], (Uint)i_m0C[l]+i_m0C_max_size*sizeof(int)-1);
  }

#if defined(EMAX7) && (defined(ARMSIML) || defined(ARMZYNQ))
  { int i;
    for (i=0; i<NLANE; i++) {
      emax7[i].dma_ctrl  = emax_info[i].dma_mmap;
      emax7[i].reg_ctrl  = emax_info[i].reg_mmap;
      ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].cmd = CMD_RESET;  // ¡ú¡ú¡ú RESET
#if defined(ARMZYNQ)
      usleep(1);
#endif
      ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].adtr = emax_info[i].ddr_mmap - emax_info[i].lmm_phys;
      ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].dmrp = 0LL;
      switch (((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].stat>>8 & 0xf) {
      case  3:EMAX_DEPTH = 64;break;
      case  2:EMAX_DEPTH = 32;break;
      case  1:EMAX_DEPTH = 16;break;
      case  0:EMAX_DEPTH =  8;break;
      default:
	printf("init_xmax: illegal stat=%x for setting EMAX_DEPTH\n",((struct reg_ctrl*)emax7[i].reg_ctrl)->i[0].stat>>8 & 0xf);
	exit(1);
      }
      printf("EMAX7[%d].DEPTH=%d\n", i, EMAX_DEPTH);
    }
  }
  printf("EMAX7: NLANE=%d DEPTH=%d\n", NLANE, EMAX_DEPTH);
#endif
}

void imemcpy(Uint *dst, Uint *src, int words)
{
  union {
    Uint i[4];
    Ull  l[2];
    Dll  d;
  } buf;

  Uint loop, i;
  if (words >= 1 && ((Ull)dst & sizeof(Uint))) { /* 4B-access odd */
    *dst++ = *src++;
    words--;
  }
  if (words >= 2 && ((Ull)dst & sizeof(Ull))) { /* 8B-access odd */
    if ((Ull)src & sizeof(Uint)) {
      buf.i[0] = *src++;
      buf.i[1] = *src++;
      *(Ull*)dst = buf.l[0];
    }
    else {
      *(Ull*)dst = *(Ull*)src;
      src += sizeof(Ull)/sizeof(Uint);
    }
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }

  if (loop = words/(sizeof(Dll)/sizeof(Uint))) {
    if ((Ull)src & sizeof(Uint)) {
      for(i=0; i<loop; i++) {
	buf.i[0] = *src++;
	buf.i[1] = *src++;
	buf.i[2] = *src++;
	buf.i[3] = *src++;
	*(Dll*)dst = buf.d;
	dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    else if ((Ull)src & sizeof(Ull)) {
      for(i=0; i<loop; i++) {
	buf.l[0] = *(Ull*)src;src += sizeof(Ull)/sizeof(Uint);
	buf.l[1] = *(Ull*)src;src += sizeof(Ull)/sizeof(Uint);
	*(Dll*)dst = buf.d;
	dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    else {
      for(i=0; i<loop; i++) {
	*(Dll*)dst = *(Dll*)src;
	src += sizeof(Dll)/sizeof(Uint);
	dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    words -= loop*(sizeof(Dll)/sizeof(Uint));
  }

  if (words >= 2) { /* 8B-access */
    if ((Ull)src & sizeof(Uint)) {
      buf.i[0] = *src++;
      buf.i[1] = *src++;
      *(Ull*)dst = buf.l[0];
    }
    else {
      *(Ull*)dst = *(Ull*)src;
      src += sizeof(Ull)/sizeof(Uint);
    }
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }
  if (words >= 1) { /* 4B-access */
    *dst++ = *src++;
    words--;
  }
}

void __attribute__((optimize("O1"))) xmax_bzero(Uint *dst, int words)
{
  /* +----+-m-----+ */
  /* |3x3 |       | */
  /* |    |    src| */
  /* +----+       | */
  /* |       +----+ */
  /* |       |    | */
  /* |       | 3x3| */
  /* +-------+----+ */
  Uint loop, i;
  if (words >= 1 && ((Ull)dst & sizeof(Uint))) { /* 4B-access odd */
    *dst++ = 0;
    words--;
  }
  if (words >= 2 && ((Ull)dst & sizeof(Ull))) { /* 8B-access odd */
    *(Ull*)dst = 0;
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }

  if (loop = words/(sizeof(Dll)/sizeof(Uint))) {
    for(i=0; i<loop; i++) {
#if __AARCH64EL__ == 1
      *((Dll*)dst) = 0;
#else
      ((Dll*)dst)->u[0] = 0;
      ((Dll*)dst)->u[1] = 0;
#endif
      dst += sizeof(Dll)/sizeof(Uint);
    }
    words -= loop*(sizeof(Dll)/sizeof(Uint));
  }

  if (words >= 2) { /* 8B-access */
    *(Ull*)dst = 0;
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }
  if (words >= 1) { /* 4B-access */
    *dst++ = 0;
    words--;
  }
}

void __attribute__((optimize("O1"))) xmax_cpyin(int order, Uint *dst, int *imo, Uint *src, int batch, int ic, int im, int m, int k)
{
  /* order 0: dst[batch][ic][im*im]  <- src[batch][ic][im*im] */
  /* order 1: dst[ic][oc][im*im]     <- src[oc][ic][im*im] */
  /* order 2: dst[ic][im][batch][im] <- src[batch][ic][im*im] */
  /* order 3: dst[im][m]             <- src[im][m]            */

  switch (order) {
  case 0:
    /* num=batch+ichan                            */
    /* imi¤Î¼þÊÕ¤Ë0¤òÄÉ²Ã¤·imo¤Ë¥³¥Ô¡¼            */
    /* k=3,(IM==M)             k=2,(IM==M)        */
    /* +-------+imo-------+    +-----+--imo----+  */
    /* | 0 0 0 |       dst|    | 0 0 |      dst|  */
    /* |  +----+im=m---+  |    |  +--+--im=m---+  */
    /* | 0|3x3 |       |  |    | 0|  |         |  */
    /* | 0|    |    src|  |    +--+--+      src|  */
    /* +--+----+       |  |    |  |            |  */
    /* |  |       +----+--+    |  |            |  */
    /* |  |       |    |0 |    |  |            |  */
    /* |  |       | 3x3|0 |    |  |            |  */
    /* |  +-------+----+  |    +--+------------+  */
    /* |          | 0 0 0 |                       */
    /* +----------+-------+                       */

    /* imi¤Èimo¤ÏÆ±¤¸¥µ¥¤¥º¤Ç¥³¥Ô¡¼                                 */
    /* k=3,(IM-k)/1+1==M       k=2,(IM-k)/1+1==M    k=1,(IM==M)     */
    /* +-------+im--------+    +-----+--im-----+                    */
    /* | x x x |       dst|    | x x |      dst|                    */
    /* |  +----+-m-----+  |    |  +--+---m-----+    +--+--im=m---+  */
    /* | x|3x3 |       |  |    | x|  |         |    |  |         |  */
    /* | x|    |    src|  |    +--+--+      src|    +--+      src|  */
    /* +--+----+       |  |    |  |            |    |            |  */
    /* |  |       +----+--+    |  |            |    |            |  */
    /* |  |       |    |x |    |  |            |    |         +--+  */
    /* |  |       | 3x3|x |    |  |            |    |         |  |  */
    /* |  +-------+----+  |    +--+------------+    +---------+--+  */
    /* |          | x x x |                                         */
    /* +----------+-------+                                         */
    /* EMAX for large IM/M                                   *//*         burst_exe 6*6    ||         burst_exe 6*6    */
    /*     +-----+  +----+-+----+---------+    +-----------+ *//* 7*8... | 7*8... | 7*8... || 7*8... | 7*8... | 7*8... */
    /* unit|2    |  |7*7 | |7*7 |*IC  *100|    |2          | *//*-- -- --                  ||-- -- --                  *//* LMM=7*8*4B */
    /*  |  |*    |  |ch0 | |ch1 |         | -> |*          | *//*         -- -- --         ||         -- -- --         *//*    =244B   */
    /*  V  |2    |  +----+ +----+         |    |2          | *//*                  -- -- --||                  -- -- --*/
    /*     |*ich |  |loop=RMGRP(6)*M(6)   |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*6*4B*4och */
    /*     +-och-+  +---------------------+    +6*6*och----+ *//* img0     img0     img0   || img1     img1     img1   *//*    =576B        */
    /*        32 ... lmf+lmxËè²óDMA            |    32/4   | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
    if (im == m && 1<k) {
      int n, i, w = im+k-1;
      for (n=0; n<batch*ic; n++,dst+=w*w,src+=im*im) {
	for (i=0; i<k/2; i++)
	  xmax_bzero(dst+i*w, w);
	for (i=k/2; i<=im+k/2-1; i++) {
	  xmax_bzero (dst+i*w,               (k/2) );
	  imemcpy(dst+i*w+(k/2),   src+(i-k/2)*im, im);
	  if (k-1-(k/2)) xmax_bzero (dst+i*w+(k/2)+im, k-1-(k/2));
	}
	for (i=im+k/2; i<w; i++)
	  xmax_bzero(dst+i*w, w);
      }
      *imo = w;
    }
    else {
      imemcpy(dst, src, batch*ic*im*im);
      *imo = im;
    }
    break;
  case 1:  /* dst[ic][oc][im*im] <- src[oc][ic][im*im] */
    {
      int i, o;
      for (o=0; o<batch; o++) {
	for (i=0; i<ic; i++)
	  imemcpy(dst+(i*batch+o)*im*im, src+(o*ic+i)*im*im, im*im);
      }
      *imo = im;
    }
    break;
  case 2:
    /* EMAX for small IM/M                                   */
    /*     +-----+  +---------------------+    +-----------+ *//*         burst_exe 6*100  ||         burst_exe 6*100  *//* 100²èÁü¤ò1Ëç(7*700pix)¤Ë(7*100¤ò7¹Ô) */
    /* unit|     |  |+----PAD----+        |    |           | *//* 7*8*100| 7*8*100| 7*8*100|| 7*8*100| 7*8*100| 7*8*100*//* ¤Þ¤¿¤Ï7*7Ï¢Â³¥¢¥É¥ì¥¹¤ò100¥»¥Ã¥È     */
    /*  |  |2    |  ||7*7 | |7*7 |*100 *IC| -> |2          | *//*-- -- --                    -- -- --                  *//* LMM=7*8*4B*100 LMMstg2-7¤Ëload       */
    /*  |  |*    |  ||im0 | |im1 |        |    |*          | *//* top=0   -- -- --            top=1   -- -- --         *//*    =22400B(RMGRP=7¤Ç2²óºÆÍøÍÑ)<32KB  */
    /*  V  |2    |  |+----+ +----+        |    |2          | *//*                  -- -- --                    -- -- --*/
    /*     |*ich |  |loop=M(6)*BATCH(100) |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*4B*100*4och */
    /*     +-och-+  +---------------------+    +6*100*och--+ *//* img0-99  img0-99  img0-99|| img0-99  img0-99  img0-99*//*    =9600B         */
    /*        32 ... lmf+lmxËè²óDMA            |      32/4 | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
    if (im == m && 1<k) {
      int n1, n0, i, w = im+k-1;
      for (n1=0; n1<batch; n1++) {           /* src-data½ç */
	for (n0=0; n0<ic; n0++,src+=im*im) { /* src-data½ç */
	  int ofs  = (n0*w*batch+n1)*w;      /* Ê£¿ôimg¤Î1¹Ô¤¬Ï¢Â³,chËè¤ËÏ¢Â³ */
	  int dist =  batch*w;               /* Ê£¿ôimg¤Î1¹Ô¤¬Ï¢Â³,»þ¥¢¥É¥ì¥¹¤Ï¼¡¹Ô */
	  for (i=0; i<k/2; i++)
	    xmax_bzero(dst+ofs+i*dist, w);
	  for (i=k/2; i<=im+k/2-1; i++) {
	    xmax_bzero (dst+ofs+i*dist,               (k/2) );
	    imemcpy(dst+ofs+i*dist+(k/2),   src+(i-k/2)*im, im);
	    if (k-1-(k/2)) xmax_bzero (dst+ofs+i*dist+(k/2)+im, k-1-(k/2));
	  }
	  for (i=im+k/2; i<w; i++)
	    xmax_bzero(dst+ofs+i*dist, w);
	}
      }
      *imo = w;
    }
    else {
      int n1, n0, i;
      for (n1=0; n1<batch; n1++) {           /* src-data½ç */
	for (n0=0; n0<ic; n0++,src+=im*im) { /* src-data½ç */
	  int ofs  = (n0*im*batch+n1)*im;
	  int dist =  batch*im;
	  for (i=0; i<im; i++)
	    imemcpy(dst+ofs+i*dist, src+i*im, im);
	}
      }
      *imo = im;
    }
    break;
  case 3:
    imemcpy(dst, src, im*m);
    *imo = im;
    break;
  }
}

void __attribute__((optimize("O1"))) xmax_cpyout(int order, Uint *dst, int batch, int oc, Uint *src, int m, int n, int oc4)
{
  /* order 0: dst[batch][oc][m*n] <- src[batch][oc4][m*n]  */
  /* order 1: dst[batch][oc][m*n] <- src[oc4][m][batch][n] */
  /* order 2: dst[m][n]           <- src[m][oc4=(n+3)&~3]  */

  /* +-dst--------------+    +-imo--------------+ */
  /* | OC | OC | OC |   | <- | OC4   | OC4   |  | */
  /* +------------------+    +------------------+ */
  int k, k2, k1, k0;

  switch (order) {
  case 0:
    for (k=0; k<batch; k++,dst+=oc*m*n,src+=oc4*m*n)
      imemcpy(dst, src, oc*m*n);
    break;
  case 1:
    for (k2=0; k2<batch; k2++) {
      for (k1=0; k1<oc; k1++) {
	for (k0=0; k0<m; k0++,dst+=n)
	  imemcpy(dst, src+((k1*m+k0)*batch+k2)*n, n);
      }
    }
    break;
  case 2:
    if (n == oc4)
      imemcpy(dst, src, m*n);
    else {
      for (k=0; k<m; k++,dst+=n,src+=oc4)
	imemcpy(dst, src, n);
    }
    break;
  }
}

/* LMM=128KB¤Î¾ì¹ç */
/* -I0 -C1 -F1 */
/*  CNN5x5  BATCH=100 M=24 RMGRP=2  IC=1  OC=16 outloop=48   inloop=2400 Klen=  400/16384 IMlen=14000/16384 Mlen=2400/8192 */
/*  GEMM00  m=100 n=10 ka=2304(/H)  8*2*3*3*16  outloop=960  inloop=15   Blen=   10/16384  Alen=11520/16384 Clen=  50/32768*/
/* -I0 -C3 -F1 */
/*  CNN5x5  BATCH=100 M=24 RMGRP=2  IC=1  OC=16 outloop=48   inloop=2400 Klen=  400/16384 IMlen=14000/16384 Mlen=2400/8192 */
/*  CNN3x3x4          M=12 RMGRP=1  IC=16 OC=16 outloop=192  inloop=1200 Klen= 2304/16384 IMlen= 4200/16384 Mlen=1200/8192 */
/*  CNN2x2            M=6  RMGRP=1  IC=16 OC=32 outloop=96   inloop=600  Klen= 1024/16384 IMlen= 1400/16384 Mlen= 600/8192 */
/*  GEMM00  m=100 n=10 ka=1152(/H)  8*2*3*3*8   outloop=240  inloop=30   Blen=   10/16384  Alen=11520/16384 Clen= 100/32768*/
/* -I1 -C4 -F1 */
/*  CNN5x5  BATCH=100 M=28 RMGRP=2  IC=3  OC=18 outloop=210  inloop=2800 Klen= 1350/16384 IMlen=16000/16384 Mlen=2800/8192 */
/*  CNN3x3x6          M=14 RMGRP=1  IC=18 OC=16 outloop=168  inloop=1400 Klen= 2592/16384 IMlen= 4800/16384 Mlen=1400/8192 */
/*  CNN2x2            M=7  RMGRP=1  IC=16 OC=32 outloop=112  inloop=700  Klen= 1024/16384 IMlen= 1600/16384 Mlen= 700/8192 */
/*  CNN2x2            M=6  RMGRP=1  IC=32 OC=64 outloop=384  inloop=600  Klen= 2048/16384 IMlen= 1400/16384 Mlen= 600/8192 */
/*  GEMM00  m=100 n=10 ka=576(/H)   8*2*3*3*4   outloop=60   inloop=60   Blen=   10/16384  Alen=11520/16384 Clen= 200/32768*/
/* -I1 -C6 -F2 */
/*  CNN5x5  BATCH=100 M=28 RMGRP=2  IC=3  OC=18 outloop=210  inloop=2800 Klen= 1350/16384 IMlen=16000/16384 Mlen=2800/8192 */
/*  CNN3x3x6          M=14 RMGRP=1  IC=18 OC=16 outloop=168  inloop=1400 Klen= 2592/16384 IMlen= 4800/16384 Mlen=1400/8192 */
/*  CNN2x2            M=7  RMGRP=1  IC=16 OC=32 outloop=112  inloop=700  Klen= 1024/16384 IMlen= 1600/16384 Mlen= 700/8192 */
/*  CNN2x2            M=6  RMGRP=1  IC=32 OC=64 outloop=384  inloop=600  Klen= 2048/16384 IMlen= 1400/16384 Mlen= 600/8192 */
/*  CNN2x2            M=2  RMGRP=1  IC=64 OC=64 outloop=256  inloop=200  Klen= 2048/16384 IMlen=  600/16384 Mlen= 200/8192 */
/*  CNN2x2            M=1  RMGRP=1  IC=64 OC=128outloop=256  inloop=100  Klen= 4096/16384 IMlen=  400/16384 Mlen= 100/8192 */
/*  GEMM00  m=100 n=40 ka=128(/H)   8*8*2       outloop=4    inloop=1000 Blen=   40/16384  Alen=12800/16384 Clen=4000/32768*/
/*  GEMM00  m=100 n=10 ka=40(/H)    8*5         outloop=1    inloop=300  Blen=   10/16384  Alen= 4000/16384 Clen=1000/32768*/

void xmax_conv_forward_5x5(int, int, float4D*, float2D*, float4D*, int);
void xmax_conv_forward_3x3x4(int, int, float4D*, float2D*, float4D*, int);
void xmax_conv_forward_3x3x6(int, int, float4D*, float2D*, float4D*, int);
void xmax_conv_forward_2x2(int, int, float4D*, float2D*, float4D*, int);

void xmax_conv_forward(int THREAD, int LANE, float4D *in, float2D *kernel, float4D *out, int ksize)
{
  switch (ksize) {
  case 5:
    xmax_conv_forward_5x5(THREAD, LANE, in, kernel, out, ksize);
    break;
  case 3:
    if (in->nchannel % 6 == 0)
      xmax_conv_forward_3x3x6(THREAD, LANE, in, kernel, out, ksize);
    else if (in->nchannel % 4 == 0)
      xmax_conv_forward_3x3x4(THREAD, LANE, in, kernel, out, ksize);
    else {
      printf("xmax_conv_forward error: ksize=%d in->nchannel=%d\n", ksize, in->nchannel);
      exit(-1);
    }
    break;
  case 2:
    xmax_conv_forward_2x2(THREAD, LANE, in, kernel, out, ksize);
    break;
  default:
    printf("xmax_conv_forward error: ksize=%d\n", ksize);
    exit(-1);
  }
}

void xmax_conv_forward_5x5(int THREAD, int LANE, float4D *in, float2D *kernel, float4D *out, int ksize)
{
  /* float4D.nstrides    .. batch_size        */
  /* float4D.nchannel    .. ichan/ochan       */
  /* float4D.kstrides    .. isize/osize       */
  /* float4D.stride_size .. isize/osize       */
  /* float4D.data                             */
  /* float2D.nstrides    .. ochan             */
  /* float2D.stride_size .. ichan*ksize*ksize */
  /* float2D.data                             */
  /* in[batch_size, ichan, isize*isize] * weight[ochan, ichan, ksize*ksize] */
  /*  -> out[batch_size, ochan, osize*osize ] */
  /* IM == M¤Î¾ì¹ç, in->data¤Î¼þÊÕ¤ËPADÄÉ²Ã   */
  /* float *i_inp; PAD+in->data ¤òcopy     */
  /* float *i_ker; ker->data    ¤òcopy     */
  /* float *i_out; out->data    ¤Øcopy     */

  /* PAD+IM*2                                 */
  /*      <-----Nich*28*28----->  <next img>  */
  /* IM A +--------++--------+ .. +--------+  */
  /*  A | | +-24-+ || +----+ | .. | +----+ |  */
  /*  | | | | ch0| || | ch1| | .. | | ch0| |  */
  /*  | | | 24   | || |    | | .. | |    | |  */
  /*  V | | +----+ || +----+ | .. | +----+ |  */
  /*    V +--------++--------+ .. +--------+  */
  /*      <PAD+IM*2>                          */
  /*        <-IM->                            */

  int   BATCH  = in->nstrides;  // 100
  int   RMGRP;
  int   IC     = in->nchannel;  // IMAP*Xin
  int   IM     = in->kstrides;  // 28
  int   OC     = out->nchannel; // W*Xout
  int   M      = out->kstrides; // 24
  int   K      = ksize;         // 5,4,3,2,1
  int   OC4    = (OC+3)&~3;
  Uint  *in0   = in->data;      // IC*IM*IM
  Uint  *ker   = kernel->data;  // OC*IC*K*K
  Uint  *out0  = out->data;     // OC*M*M
  Uint  *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *kp,  kidx, *op;
  int   pad;
  int   count, top, iset, oc, w, ic, y, x;
  Ull   IM4, M4, IM4M4, IMlen, IMlenK, Klen, Mlen;
  Ull   CHIP, img, rofs, cofs, iofs, oofs, k;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;
  Ull   Force1, Force2;

  if (IM == M)
    pad = 0;   /* PADÌµ¤·.in¼þ°Ï0.0¤ò²¾Äê */
  else if ((IM - K)/1 + 1 == M)
    pad = K/2; /* PADÍ­¤ê.in¼þ°ÏÆÃÊÌ°·¤¤ÉÔÍ× */
  else {
    printf("xmax_conv_forward_5x5 error: IM=%d K=%d M=%d\n", IM, K, M);
    printf("IM == M || (IM-K)/1+1 == M\n");
    exit(-1);
  }

  /* i_inp, i_ker, i_out¤Ï³ÎÊÝºÑ¤À¤¬À­Ç½É¾²Á¤Ë¤Ï»È¤ï¤Ê¤¤ */
  /*printf("<<<XMAX(C)>>>\n");*/
  /*printf("xmax IM=%d M=%d K=%d %d*%d*%d\n", IM, M, K, OC, BATCH*M*M, IC*K*K);*/
  /*printf("<<<XMAX(REAL)>>>\n");*/

#if 0
    RMGRP = M; /* RMGRP = 24 {28,1,5,24,9,2} */
               /* RMGRP = 28 {32,3,5,28,11,2}*/
#undef  IMAP
#undef  W
#undef  NCHIP
/* IMAP > 1 ¡ú¡ú¡ú PBL1-1 ¡ú¡ú¡ú */
/* NCHIP  4 ¡ú¡ú¡ú PBL1-1 ¡ú¡ú¡ú */
#define IMAP  1
#define W     4
#define NCHIP 1
    /*{28,1,5,24,9,2}/{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,7,32,1},{7,32,2,6, 32,2}*/
    /* AAAAAAAAAAAAA / AAAAAAAAAAAAAA                                                    */
    monitor_time_start(THREAD, IMAX_CPYIN);
    xmax_cpyin(0, i_inp[LANE], &IM, in0, BATCH, IC, IM, M, K); /* ¤³¤Î»þÅÀ¤Ç0.0¤ÎPAD¤òÄÉ²Ã¤Ç¤­¤ë */
    xmax_cpyin(0, i_ker[LANE], &K,  ker, OC,    IC,  K, K, 1); /* ½ÐÎÏ¤Î¤ß²ó¼ý¤¹¤ì¤Ð¤è¤¤ */
    xmax_bzero(i_out[LANE], BATCH*OC4*M*M);
    monitor_time_end(THREAD, IMAX_CPYIN);
    IM4   = IM*4;
    M4    = M*4;
    IM4M4 = IM4<<32|M4;
    IMlen = IM*(RMGRP+(K-1));
    Klen  = OC*IC*K*K;
    Mlen  = M*RMGRP;
    Force1 = 1;

    if (Klen > LMEM_SIZE/4/2 || IMlen > LMEM_SIZE/4/2 || Mlen > LMEM_SIZE/4/4)
      printf("   CNN5x5  BATCH=%d M=%d RMGRP=%d IC=%d OC=%d outloop[BATCH*M/RMGRP*IC/IMAP*OC4/NCHIP/W]=%d inloop[RMGRP*M]=%d Klen=%d/%d IMlen=%d/%d Mlen=%d/%d\n",
	     (Uint)BATCH, (Uint)M, (Uint)RMGRP, (Uint)IC, (Uint)OC, (Uint)(BATCH*M/RMGRP*IC/IMAP*OC4/NCHIP/W), (Uint)(RMGRP*M), (Uint)Klen, LMEM_SIZE/4/2, (Uint)IMlen, LMEM_SIZE/4/2, (Uint)Mlen, LMEM_SIZE/4/4);

    for (img=0; img<BATCH; img++) {
      for (top=0; top<M; top+=RMGRP) {
        for (iset=0; iset<IC; iset+=IMAP) {  /* accumulate multiple sets of IC */
          Uint *ip0 = &i_inp[LANE][(img*IC+iset+0)*IM*IM]; /* top of input#0 */
          Uint *it0 = ip0+top*IM, *ip00[25];
	  ip00[ 0] = ip0+(top+0)*IM+0; ip00[ 1] = ip0+(top+0)*IM+1; ip00[ 2] = ip0+(top+0)*IM+2; ip00[ 3] = ip0+(top+0)*IM+3; ip00[ 4] = ip0+(top+0)*IM+4;
	  ip00[ 5] = ip0+(top+1)*IM+0; ip00[ 6] = ip0+(top+1)*IM+1; ip00[ 7] = ip0+(top+1)*IM+2; ip00[ 8] = ip0+(top+1)*IM+3; ip00[ 9] = ip0+(top+1)*IM+4;
	  ip00[10] = ip0+(top+2)*IM+0; ip00[11] = ip0+(top+2)*IM+1; ip00[12] = ip0+(top+2)*IM+2; ip00[13] = ip0+(top+2)*IM+3; ip00[14] = ip0+(top+2)*IM+4;
	  ip00[15] = ip0+(top+3)*IM+0; ip00[16] = ip0+(top+3)*IM+1; ip00[17] = ip0+(top+3)*IM+2; ip00[18] = ip0+(top+3)*IM+3; ip00[19] = ip0+(top+3)*IM+4;
	  ip00[20] = ip0+(top+4)*IM+0; ip00[21] = ip0+(top+4)*IM+1; ip00[22] = ip0+(top+4)*IM+2; ip00[23] = ip0+(top+4)*IM+3; ip00[24] = ip0+(top+4)*IM+4;

          for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */
            Uint *kp00[NCHIP],*kp01[NCHIP],*kp02[NCHIP],*kp03[NCHIP];
            Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP];
            Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP];

            for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
              Uint choc  = CHIP*OC4/NCHIP+oc;
              kp00[CHIP]= (choc+0<OC) ? i_ker[LANE]+((choc+0)*IC+iset+0)*K*K : i_ker[LANE];
	      kp01[CHIP]= (choc+1<OC) ? i_ker[LANE]+((choc+1)*IC+iset+0)*K*K : i_ker[LANE];
	      kp02[CHIP]= (choc+2<OC) ? i_ker[LANE]+((choc+2)*IC+iset+0)*K*K : i_ker[LANE];
	      kp03[CHIP]= (choc+3<OC) ? i_ker[LANE]+((choc+3)*IC+iset+0)*K*K : i_ker[LANE];
              op0[CHIP] = i_out[LANE]+(img*OC4+choc+0)*M*M+top*M; op1[CHIP] = i_out[LANE]+(img*OC4+choc+1)*M*M+top*M; op2[CHIP] = i_out[LANE]+(img*OC4+choc+2)*M*M+top*M; op3[CHIP] = i_out[LANE]+(img*OC4+choc+3)*M*M+top*M;
              ot0[CHIP] = i_out[LANE]+(img*OC4+choc+0)*M*M+top*M; ot1[CHIP] = i_out[LANE]+(img*OC4+choc+1)*M*M+top*M; ot2[CHIP] = i_out[LANE]+(img*OC4+choc+2)*M*M+top*M; ot3[CHIP] = i_out[LANE]+(img*OC4+choc+3)*M*M+top*M;
            }
	    Force2 = 1;

#define cnn5x5_core1(b, o, bp1, n, Force) \
  mop(OP_LDWR,   1, &BR[b][0][1],  (Ull)kp00[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][0][0],  (Ull)kp01[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][1],  (Ull)kp02[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][0],  (Ull)kp03[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][2][1],  (Ull)ip00[n], iofs, MSK_W1, (Ull)it0, IMlen, 0, 0, (Ull)NULL, IMlen);\
  exe(OP_FMA, &AR[bp1][0], AR[b][0], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][1], AR[b][1], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][2], AR[b][2], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][3], AR[b][3], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn5x5_final(b, bp1, Force) \
  mop(OP_LDWR,   1, &BR[bp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  exe(OP_FAD, &AR[bp1][0], AR[b][0], EXP_H3210, BR[bp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][1], AR[b][1], EXP_H3210, BR[bp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][2], AR[b][2], EXP_H3210, BR[bp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][3], AR[b][3], EXP_H3210, BR[bp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  mop(OP_STWR,   1, &AR[bp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen)

//EMAX5A begin cnn5x5 mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
        /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-IM4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                      /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,    &rofs, rofs,            EXP_H3210, INIT0?IM4M4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &iofs, rofs,            EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
                  exe(OP_ADD,    &oofs, rofs,            EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

                  /****in0*****/
                  mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp00[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp01[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp02[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp03[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 10KB */
                  mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)ip00[0],   iofs, MSK_W1, (Ull)it0,         IMlen,0, 0,      (Ull)NULL, IMlen);/* stage#2 10KB */
                  exe(OP_FML, &AR[3][0], BR[2][2][1], EXP_H3210, BR[2][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][1], BR[2][2][1], EXP_H3210, BR[2][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][2], BR[2][2][1], EXP_H3210, BR[2][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][3], BR[2][2][1], EXP_H3210, BR[2][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		  cnn5x5_core1( 3, 4LL, 4, 1, Force1);
		  cnn5x5_core1( 4, 8LL, 5, 2, Force1);
		  cnn5x5_core1( 5,12LL, 6, 3, Force1);
		  cnn5x5_core1( 6,16LL, 7, 4, Force1);
		  cnn5x5_core1( 7,20LL, 8, 5, Force1);
		  cnn5x5_core1( 8,24LL, 9, 6, Force1);
		  cnn5x5_core1( 9,28LL,10, 7, Force1);
		  cnn5x5_core1(10,32LL,11, 8, Force1);
		  cnn5x5_core1(11,36LL,12, 9, Force1);
		  cnn5x5_core1(12,40LL,13,10, Force1);
		  cnn5x5_core1(13,44LL,14,11, Force1);
		  cnn5x5_core1(14,48LL,15,12, Force1);
		  cnn5x5_core1(15,52LL,16,13, Force1);
		  cnn5x5_core1(16,56LL,17,14, Force1);
		  cnn5x5_core1(17,60LL,18,15, Force1);
		  cnn5x5_core1(18,64LL,19,16, Force1);
		  cnn5x5_core1(19,68LL,20,17, Force1);
		  cnn5x5_core1(20,72LL,21,18, Force1);
		  cnn5x5_core1(21,76LL,22,19, Force1);
		  cnn5x5_core1(22,80LL,23,20, Force1);
		  cnn5x5_core1(23,84LL,24,21, Force1);
		  cnn5x5_core1(24,88LL,25,22, Force1);
		  cnn5x5_core1(25,92LL,26,23, Force1);
		  cnn5x5_core1(26,96LL,27,24, Force1);
                  /****final*****/
		  cnn5x5_final(27,     28, Force2);
                }
              }
            }
//EMAX5A end
            if (Force1) Force1 = 0;
            if (Force2) Force2 = 0;
printf("5");
          }
        }
      }
    }
//EMAX5A drain_dirty_lmm
    monitor_time_start(THREAD, IMAX_CPYOUT); 
    xmax_cpyout(0, out0, BATCH, OC, i_out[LANE], M, M, OC4);
    monitor_time_end(THREAD, IMAX_CPYOUT); 
#else
    RMGRP = 2; /* RMGRP = 7 /*{7,16,2,7,32,1}*/
               /* RMGRP = 6 /*{7,32,2,6,32,2}*/
    /* CIFAR IMAGE                                           */
    /* +--------------------+------------                    */
    /* |W*W(R) W*W(G) W*W(B)|..batch_size                    */
    /* +--------------------+------------                    */
    /*                                                       */
    /* GPU ORIGINAL                                          */
    /*     +-2*2*ich-+  +-100*6*6-----+        +-100*6*6---+ */
    /* och0|         |  |2            |    och0|           | */
    /* och1|         |  |* 7*7¤òunpack| -> och1|           | */
    /* och2|         |  |2            |    och2|           | */
    /*     +---------+  |*ich         |        +-----------+ */
    /*                  +-------------+                      */
    /* EMAX ORIGINAL                                         *//*         burst_exe 6*6    ||         burst_exe 6*6    */
    /*     +-----+  +----+-+----+---------+    +-----------+ *//* 7*8... | 7*8... | 7*8... || 7*8... | 7*8... | 7*8... */
    /* unit|2    |  |7*7 | |7*7 |*IC  *100|    |2          | *//*-- -- --                  ||-- -- --                  *//* LMM=7*8*4B */
    /*  |  |*    |  |ch0 | |ch1 |         | -> |*          | *//*         -- -- --         ||         -- -- --         *//*    =244B   */
    /*  V  |2    |  +----+ +----+         |    |2          | *//*                  -- -- --||                  -- -- --*/
    /*     |*ich |  |loop=RMGRP(6)*M(6)   |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*6*4B*4och */
    /*     +-och-+  +---------------------+    +6*6*och----+ *//* img0     img0     img0   || img1     img1     img1   *//*    =576B        */
    /*        32 ... lmf+lmxËè²óDMA            |    32/4   | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
    /* EMAX for small IM/M                                    */
    /*     +-----+  +---------------------+    +-----------+ *//*         burst_exe 6*100                              */
    /* unit|     |  |+----PAD----+        |    |           | *//* 7*2... | 7*2... | 7*2... || 7*2... | 7*2... | 7*2..  */
    /*  |  |2    |  ||7*7 | |7*7 |*IC *100| -> |2          | *//* -                          -                         *//* LMM=7*8*4B*32ch*100 */
    /*  |  |*    |  ||ch0 | |ch1 |        |    |*          | *//*          -                          -                *//*    =716800B         */
    /*  V  |2    |  |+----+ +----+        |    |2          | *//*                   -                          -       */
    /*     |*ich |  |loop=M(6)*BATCH(100) |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*6*4B*100*4och */
    /*     +-och-+  +---------------------+    +6*100*och--+ *//* img0     img0     img0   || img1     img1     img1   *//*    =57600B          */
    /*        32 ... lmf+lmxËè²óDMA            |      32/4 | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
    /* EMAX for small IM/M                                   */
    /*     +-----+  +---------------------+    +-----------+ *//*         burst_exe 6*100  ||         burst_exe 6*100  *//* 100²èÁü¤ò1Ëç(7*700pix)¤Ë(7*100¤ò7¹Ô) */
    /* unit|     |  |+----PAD----+        |    |           | *//* 7*8*100| 7*8*100| 7*8*100|| 7*8*100| 7*8*100| 7*8*100*//* ¤Þ¤¿¤Ï7*7Ï¢Â³¥¢¥É¥ì¥¹¤ò100¥»¥Ã¥È     */
    /*  |  |2    |  ||7*7 | |7*7 |*100 *IC| -> |2          | *//*-- -- --                    -- -- --                  *//* LMM=7*8*4B*100 LMMstg2-7¤Ëload       */
    /*  |  |*    |  ||im0 | |im1 |        |    |*          | *//* top=0   -- -- --            top=1   -- -- --         *//*    =22400B(RMGRP=7¤Ç2²óºÆÍøÍÑ)<32KB  */
    /*  V  |2    |  |+----+ +----+        |    |2          | *//*                  -- -- --                    -- -- --*/
    /*     |*ich |  |loop=M(6)*BATCH(100) |    |*ich       | *//* stg2     stg4     stg6   || stg2     stg4     stg6   *//* out=6*4B*100*4och */
    /*     +-och-+  +---------------------+    +6*100*och--+ *//* img0-99  img0-99  img0-99|| img0-99  img0-99  img0-99*//*    =9600B         */
    /*        32 ... lmf+lmxËè²óDMA            |      32/4 | *//* ch0      ch1      ch2    || ch0      ch1      ch2    */
    /*                                         +-----------+ */
#undef  IMAP
#undef  W
#undef  NCHIP
/* IMAP > 1 ¡ú¡ú¡ú PBL1-1 ¡ú¡ú¡ú */
/* NCHIP  4 ¡ú¡ú¡ú PBL1-1 ¡ú¡ú¡ú */
#define IMAP  1
#define W     4
#define NCHIP 1
    /*{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,7,32,1},{7,32,2,6,32,2}*/
    /*                                    AAAAAAAAAAAAA   AAAAAAAAAAAAA */
    monitor_time_start(THREAD, IMAX_CPYIN); 
    xmax_cpyin(2, i_inp[LANE], &IM, in0, BATCH, IC, IM, M, K); /* ¤³¤Î»þÅÀ¤Ç0.0¤ÎPAD¤òÄÉ²Ã¤Ç¤­¤ë */
    xmax_cpyin(0, i_ker[LANE], &K,  ker, OC,    IC,  K, K, 1); /* ½ÐÎÏ¤Î¤ß²ó¼ý¤¹¤ì¤Ð¤è¤¤ */
    xmax_bzero(i_out[LANE], BATCH*OC4*M*M);
    monitor_time_end(THREAD, IMAX_CPYIN); 
    IM4   = IM*4;
    M4    = M*4;
    IM4M4 = IM4<<32|M4;
    IMlenK= IM*BATCH*K;
    IMlen = IM*BATCH;
    Klen  = OC*IC*K*K;
    Mlen  = M*BATCH;
    Force1 = 1;

    if (Klen > LMEM_SIZE/4/2 || IMlenK > LMEM_SIZE/4/2 || Mlen > LMEM_SIZE/4/4)
      printf("   CNN5x5  M=%d RMGRP=%d IC=%d OC=%d outloop[M/RMGRP*IC/IMAP*OC4/NCHIP/W]=%d inloop[BATCH*M]=%d Klen=%d/%d IMlenK=%d/%d Mlen=%d/%d\n",
	     (Uint)M, (Uint)RMGRP, (Uint)IC, (Uint)OC, (Uint)(M/RMGRP*IC/IMAP*OC4/NCHIP/W), (Uint)(BATCH*M), (Uint)Klen, LMEM_SIZE/4/2, (Uint)IMlenK, LMEM_SIZE/4/2, (Uint)Mlen, LMEM_SIZE/4/4);

    for (top=0; top<M; top+=RMGRP) { /* divide M into two */
      for (iset=0; iset<IC; iset+=IMAP) {  /* accumulate multiple sets of IC */
	Uint *ip0 = &i_inp[LANE][iset*IM*BATCH*IM]; /* top of input#0 */
	Uint *it0 = ip0+ top   *IM*BATCH;
	Uint *it1 = ip0+(top+K)*IM*BATCH;
	Uint *ip00[(K+RMGRP)*K];
	for (rofs=0; rofs<K+RMGRP; rofs++) {
	  for (k=0; k<K; k++)
	    ip00[K*rofs+k] = ip0+(top+rofs)*IM*BATCH+k;
	}

	for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */
	  Uint *kp00[NCHIP],*kp01[NCHIP],*kp02[NCHIP],*kp03[NCHIP];
	  Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP], *op4[NCHIP], *op5[NCHIP], *op6[NCHIP], *op7[NCHIP];
	  Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP], *ot4[NCHIP], *ot5[NCHIP], *ot6[NCHIP], *ot7[NCHIP];

	  for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
	    Uint choc  = CHIP*OC4/NCHIP+oc;
	    kp00[CHIP] = (choc+0<OC) ? i_ker[LANE]+((choc+0)*IC+iset+0)*K*K : i_ker[LANE];
	    kp01[CHIP] = (choc+1<OC) ? i_ker[LANE]+((choc+1)*IC+iset+0)*K*K : i_ker[LANE];
	    kp02[CHIP] = (choc+2<OC) ? i_ker[LANE]+((choc+2)*IC+iset+0)*K*K : i_ker[LANE];
	    kp03[CHIP] = (choc+3<OC) ? i_ker[LANE]+((choc+3)*IC+iset+0)*K*K : i_ker[LANE];
	    op0[CHIP] = ot0[CHIP] = i_out[LANE]+((choc+0)*M+top  )*M*BATCH;
	    op1[CHIP] = ot1[CHIP] = i_out[LANE]+((choc+1)*M+top  )*M*BATCH;
	    op2[CHIP] = ot2[CHIP] = i_out[LANE]+((choc+2)*M+top  )*M*BATCH;
	    op3[CHIP] = ot3[CHIP] = i_out[LANE]+((choc+3)*M+top  )*M*BATCH;
	    op4[CHIP] = ot4[CHIP] = i_out[LANE]+((choc+0)*M+top+1)*M*BATCH;
	    op5[CHIP] = ot5[CHIP] = i_out[LANE]+((choc+1)*M+top+1)*M*BATCH;
	    op6[CHIP] = ot6[CHIP] = i_out[LANE]+((choc+2)*M+top+1)*M*BATCH;
	    op7[CHIP] = ot7[CHIP] = i_out[LANE]+((choc+3)*M+top+1)*M*BATCH;
	  }
	  Force2 = 1;

#define cnn5x5_core1(b, o, bp1, n, Force) \
  mop(OP_LDWR,   1, &BR[b][0][1],  (Ull)kp00[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][0][0],  (Ull)kp01[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][1],  (Ull)kp02[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][0],  (Ull)kp03[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][2][1],  (Ull)ip00[n], iofs, MSK_W1, (Ull)it0,        IMlenK,0, 0,     (Ull)NULL, IMlenK);\
  exe(OP_FMA, &AR[bp1][0], AR[b][0], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][1], AR[b][1], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][2], AR[b][2], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][3], AR[b][3], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn5x5_core2(b, o, bp1, n, Force) \
  mop(OP_LDWR,   1, &BR[b][0][1],  (Ull)kp00[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][0][0],  (Ull)kp01[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][1],  (Ull)kp02[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][0],  (Ull)kp03[CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][2][1],  (Ull)ip00[n], iofs, MSK_W1, (Ull)it1,         IMlen,0, 0,     (Ull)NULL, IMlen);\
  exe(OP_FMA, &AR[bp1][0], AR[b][0], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][1], AR[b][1], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][2], AR[b][2], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][3], AR[b][3], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn5x5_fin1(b, bp1, Force) \
  mop(OP_LDWR,   1, &BR[bp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  exe(OP_FAD, &AR[bp1][0], AR[b][0], EXP_H3210, BR[bp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][1], AR[b][1], EXP_H3210, BR[bp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][2], AR[b][2], EXP_H3210, BR[bp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][3], AR[b][3], EXP_H3210, BR[bp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  mop(OP_STWR,   1, &AR[bp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen)

#define cnn5x5_fin2(b, bp1, Force) \
  mop(OP_LDWR,   1, &BR[bp1][0][1],  (Ull)op4[CHIP], oofs, MSK_W0, (Ull)ot4[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][1][1],  (Ull)op5[CHIP], oofs, MSK_W0, (Ull)ot5[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][2][1],  (Ull)op6[CHIP], oofs, MSK_W0, (Ull)ot6[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][3][1],  (Ull)op7[CHIP], oofs, MSK_W0, (Ull)ot7[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  exe(OP_FAD, &AR[bp1][0], AR[b][0], EXP_H3210, BR[bp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][1], AR[b][1], EXP_H3210, BR[bp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][2], AR[b][2], EXP_H3210, BR[bp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][3], AR[b][3], EXP_H3210, BR[bp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  mop(OP_STWR,   1, &AR[bp1][0], oofs, (Ull)op4[CHIP], MSK_D0, (Ull)ot4[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][1], oofs, (Ull)op5[CHIP], MSK_D0, (Ull)ot5[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][2], oofs, (Ull)op6[CHIP], MSK_D0, (Ull)ot6[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][3], oofs, (Ull)op7[CHIP], MSK_D0, (Ull)ot7[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen)

//EMAX5A begin cnn5x5 mapdist=0
    /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
      /*2*/ for (INIT1=1,LOOP1=BATCH,img=(0-IM4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                       /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
        /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                exe(OP_ADD,    &img,  img,             EXP_H3210, INIT0?IM4M4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
                exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                exe(OP_ADD,    &iofs, img,             EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
                exe(OP_ADD,    &oofs, img,             EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

                /****core1*****/
                mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp00[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp01[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp02[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp03[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 10KB */
                mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)ip00[0],   iofs, MSK_W1, (Ull)it0,        IMlenK,0, 0,      (Ull)NULL, IMlenK);/* stage#2 10KB */
                exe(OP_FML, &AR[3][0], BR[2][2][1], EXP_H3210, BR[2][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#3 */
                exe(OP_FML, &AR[3][1], BR[2][2][1], EXP_H3210, BR[2][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#3 */
                exe(OP_FML, &AR[3][2], BR[2][2][1], EXP_H3210, BR[2][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#3 */
                exe(OP_FML, &AR[3][3], BR[2][2][1], EXP_H3210, BR[2][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#3 */
		cnn5x5_core1( 3, 4LL, 4, 1, Force1);
		cnn5x5_core1( 4, 8LL, 5, 2, Force1);
		cnn5x5_core1( 5,12LL, 6, 3, Force1);
		cnn5x5_core1( 6,16LL, 7, 4, Force1);
		cnn5x5_core1( 7,20LL, 8, 5, Force1);
		cnn5x5_core1( 8,24LL, 9, 6, Force1);
		cnn5x5_core1( 9,28LL,10, 7, Force1);
		cnn5x5_core1(10,32LL,11, 8, Force1);
		cnn5x5_core1(11,36LL,12, 9, Force1);
		cnn5x5_core1(12,40LL,13,10, Force1);
		cnn5x5_core1(13,44LL,14,11, Force1);
		cnn5x5_core1(14,48LL,15,12, Force1);
		cnn5x5_core1(15,52LL,16,13, Force1);
		cnn5x5_core1(16,56LL,17,14, Force1);
		cnn5x5_core1(17,60LL,18,15, Force1);
		cnn5x5_core1(18,64LL,19,16, Force1);
		cnn5x5_core1(19,68LL,20,17, Force1);
		cnn5x5_core1(20,72LL,21,18, Force1);
		cnn5x5_core1(21,76LL,22,19, Force1);
		cnn5x5_core1(22,80LL,23,20, Force1);
		cnn5x5_core1(23,84LL,24,21, Force1);
		cnn5x5_core1(24,88LL,25,22, Force1);
		cnn5x5_core1(25,92LL,26,23, Force1);
		cnn5x5_core1(26,96LL,27,24, Force1);
                /****final*****/
		cnn5x5_fin1(27,     28, Force2);

                /****core2*****/
                mop(OP_LDWR,   1, &BR[29][0][1],  (Ull)kp00[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#29 */
                mop(OP_LDWR,   1, &BR[29][0][0],  (Ull)kp01[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#29 */
                mop(OP_LDWR,   1, &BR[29][1][1],  (Ull)kp02[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#29 */
                mop(OP_LDWR,   1, &BR[29][1][0],  (Ull)kp03[CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#29 10KB */
                mop(OP_LDWR,   1, &BR[29][2][1],  (Ull)ip00[5],   iofs, MSK_W1, (Ull)it0,        IMlenK,0, 0,     (Ull)NULL, IMlenK);/* stage#29 10KB */
                exe(OP_FML, &AR[30][0], BR[29][2][1], EXP_H3210, BR[29][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
                exe(OP_FML, &AR[30][1], BR[29][2][1], EXP_H3210, BR[29][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
                exe(OP_FML, &AR[30][2], BR[29][2][1], EXP_H3210, BR[29][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
                exe(OP_FML, &AR[30][3], BR[29][2][1], EXP_H3210, BR[29][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
		cnn5x5_core1(30, 4LL,31, 6, Force1);
		cnn5x5_core1(31, 8LL,32, 7, Force1);
		cnn5x5_core1(32,12LL,33, 8, Force1);
		cnn5x5_core1(33,16LL,34, 9, Force1);
		cnn5x5_core1(34,20LL,35,10, Force1);
		cnn5x5_core1(35,24LL,36,11, Force1);
		cnn5x5_core1(36,28LL,37,12, Force1);
		cnn5x5_core1(37,32LL,38,13, Force1);
		cnn5x5_core1(38,36LL,39,14, Force1);
		cnn5x5_core1(39,40LL,40,15, Force1);
		cnn5x5_core1(40,44LL,41,16, Force1);
		cnn5x5_core1(41,48LL,42,17, Force1);
		cnn5x5_core1(42,52LL,43,18, Force1);
		cnn5x5_core1(43,56LL,44,19, Force1);
		cnn5x5_core1(44,60LL,45,20, Force1);
		cnn5x5_core1(45,64LL,46,21, Force1);
		cnn5x5_core1(46,68LL,47,22, Force1);
		cnn5x5_core1(47,72LL,48,23, Force1);
		cnn5x5_core1(48,76LL,49,24, Force1);
		cnn5x5_core2(49,80LL,50,25, Force1);
		cnn5x5_core2(50,84LL,51,26, Force1);
		cnn5x5_core2(51,88LL,52,27, Force1);
		cnn5x5_core2(52,92LL,53,28, Force1);
		cnn5x5_core2(53,96LL,54,29, Force1);
                /****final*****/
		cnn5x5_fin2(54,     55, Force2);
              }
            }
          }
//EMAX5A end
          if (Force1) Force1 = 0;
          if (Force2) Force2 = 0;
printf("5");
        }
      }
    }
//EMAX5A drain_dirty_lmm
    monitor_time_start(THREAD, IMAX_CPYOUT); 
    xmax_cpyout(1, out0, BATCH, OC, i_out[LANE], M, M, OC4);
    monitor_time_end(THREAD, IMAX_CPYOUT); 
#endif
}

void xmax_conv_forward_3x3x4(int THREAD, int LANE, float4D *in, float2D *kernel, float4D *out, int ksize)
{
  int   BATCH  = in->nstrides;  // 100
  int   RMGRP;
  int   IC     = in->nchannel;  // IMAP*Xin
  int   IM     = in->kstrides;  // 28
  int   OC     = out->nchannel; // W*Xout
  int   M      = out->kstrides; // 24
  int   K      = ksize;         // 5,4,3,2,1
  int   OC4    = (OC+3)&~3;
  Uint  *in0   = in->data;      // IC*IM*IM
  Uint  *ker   = kernel->data;  // OC*IC*K*K
  Uint  *out0  = out->data;     // OC*M*M
  Uint  *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *kp,  kidx, *op;
  int   pad;
  int   count, top, iset, oc, w, ic, y, x;
  Ull   IM4, M4, IM4M4, IMlen, Klen, Mlen;
  Ull   CHIP, img, rofs, cofs, iofs, oofs, k;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;
  Ull   Force1, Force2;

  if (IM == M)
    pad = 0;   /* PADÌµ¤·.in¼þ°Ï0.0¤ò²¾Äê */
  else if ((IM - K)/1 + 1 == M)
    pad = K/2; /* PADÍ­¤ê.in¼þ°ÏÆÃÊÌ°·¤¤ÉÔÍ× */
  else {
    printf("xmax_conv_forward_3x3x4 error: IM=%d K=%d M=%d\n", IM, K, M);
    printf("IM == M || (IM-K)/1+1 == M\n");
    exit(-1);
  }

  /* i_inp, i_ker, i_out¤Ï³ÎÊÝºÑ¤À¤¬À­Ç½É¾²Á¤Ë¤Ï»È¤ï¤Ê¤¤ */
  /*printf("<<<XMAX(C)>>>\n");*/
  /*printf("xmax IM=%d M=%d K=%d %d*%d*%d\n", IM, M, K, OC, BATCH*M*M, IC*K*K);*/
  /*printf("<<<XMAX(REAL)>>>\n");*/

    RMGRP = 1; /* RMGRP = 7 /*{7,16,2,7,32,1}*/
               /* RMGRP = 6 /*{7,32,2,6,32,2}*/
#undef  IMAP
#undef  W
#undef  NCHIP
/* IMAP > 1 ¡ú¡ú¡ú PBL1-1 ¡ú¡ú¡ú */
/* NCHIP  4 ¡ú¡ú¡ú PBL1-1 ¡ú¡ú¡ú */
#define IMAP  4
#define W     4
#define NCHIP 1
    /*{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,7,32,1},{7,32,2,6,32,2}*/
    /*                                    AAAAAAAAAAAAA   AAAAAAAAAAAAA */
    monitor_time_start(THREAD, IMAX_CPYIN); 
    xmax_cpyin(2, i_inp[LANE], &IM, in0, BATCH, IC, IM, M, K); /* ¤³¤Î»þÅÀ¤Ç0.0¤ÎPAD¤òÄÉ²Ã¤Ç¤­¤ë */
    xmax_cpyin(0, i_ker[LANE], &K,  ker, OC,    IC,  K, K, 1); /* ½ÐÎÏ¤Î¤ß²ó¼ý¤¹¤ì¤Ð¤è¤¤ */
    xmax_bzero(i_out[LANE], BATCH*OC4*M*M);
    monitor_time_end(THREAD, IMAX_CPYIN); 
    IM4   = IM*4;
    M4    = M*4;
    IM4M4 = IM4<<32|M4;
    IMlen = IM*BATCH*(RMGRP+(K-1));
    Klen  = OC*IC*K*K;
    Mlen  = M*BATCH*RMGRP;
    Force1 = 1;

    if (Klen > LMEM_SIZE/4/2 || IMlen > LMEM_SIZE/4/2 || Mlen > LMEM_SIZE/4/4)
      printf("   CNN3x3x4  M=%d RMGRP=%d IC=%d OC=%d outloop[M/RMGRP*IC/IMAP*OC4/NCHIP/W]=%d inloop[BATCH*M]=%d Klen=%d/%d IMlen=%d/%d Mlen=%d/%d\n",
	     (Uint)M, (Uint)RMGRP, (Uint)IC, (Uint)OC, (Uint)(M/RMGRP*IC/IMAP*OC4/NCHIP/W), (Uint)(BATCH*M), (Uint)Klen, LMEM_SIZE/4/2, (Uint)IMlen, LMEM_SIZE/4/2, (Uint)Mlen, LMEM_SIZE/4/4);

    for (top=0; top<M; top+=RMGRP) {
      for (iset=0; iset<IC; iset+=IMAP) {  /* accumulate multiple sets of IC */
	Uint *ip[IMAP], *it[IMAP], *ip0[IMAP][9];
	for (k=0; k<IMAP; k++) {
	  ip[k]     = &i_inp[LANE][(iset+k)*IM*BATCH*IM]; /* top of input#0 */
	  it[k]     = top*IM*BATCH+ip[k];
	  ip0[k][0] = ip[k]+(top+0)*IM*BATCH+0; ip0[k][1] = ip[k]+(top+0)*IM*BATCH+1; ip0[k][2] = ip[k]+(top+0)*IM*BATCH+2;
	  ip0[k][3] = ip[k]+(top+1)*IM*BATCH+0; ip0[k][4] = ip[k]+(top+1)*IM*BATCH+1; ip0[k][5] = ip[k]+(top+1)*IM*BATCH+2;
	  ip0[k][6] = ip[k]+(top+2)*IM*BATCH+0; ip0[k][7] = ip[k]+(top+2)*IM*BATCH+1; ip0[k][8] = ip[k]+(top+2)*IM*BATCH+2;
	}

        for (rofs=0; rofs<RMGRP&&(top+rofs)<M; rofs++) { /* image loop (row) */

	  for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */
	    Uint *kp0[IMAP][NCHIP],*kp1[IMAP][NCHIP],*kp2[IMAP][NCHIP],*kp3[IMAP][NCHIP];
	    Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP];
	    Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP];

            for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
              Uint choc  = CHIP*OC4/NCHIP+oc;
	      for (k=0; k<IMAP; k++) {
		kp0[k][CHIP] = (choc+0<OC) ? i_ker[LANE]+((choc+0)*IC+iset+k+0)*K*K : i_ker[LANE];
		kp1[k][CHIP] = (choc+1<OC) ? i_ker[LANE]+((choc+1)*IC+iset+k+0)*K*K : i_ker[LANE];
		kp2[k][CHIP] = (choc+2<OC) ? i_ker[LANE]+((choc+2)*IC+iset+k+0)*K*K : i_ker[LANE];
		kp3[k][CHIP] = (choc+3<OC) ? i_ker[LANE]+((choc+3)*IC+iset+k+0)*K*K : i_ker[LANE];
	      }
              op0[CHIP] = i_out[LANE]+((choc+0)*M+top)*M*BATCH; op1[CHIP] = i_out[LANE]+((choc+1)*M+top)*M*BATCH; op2[CHIP] = i_out[LANE]+((choc+2)*M+top)*M*BATCH; op3[CHIP] = i_out[LANE]+((choc+3)*M+top)*M*BATCH;
              ot0[CHIP] = i_out[LANE]+((choc+0)*M+top)*M*BATCH; ot1[CHIP] = i_out[LANE]+((choc+1)*M+top)*M*BATCH; ot2[CHIP] = i_out[LANE]+((choc+2)*M+top)*M*BATCH; ot3[CHIP] = i_out[LANE]+((choc+3)*M+top)*M*BATCH;
            }
	    Force2 = 1;

#define cnn3x3x4_core1(b, i, o, bp1, n, Force) \
  mop(OP_LDWR,   1, &BR[b][0][1],  (Ull)kp0[i][CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][0][0],  (Ull)kp1[i][CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][1],  (Ull)kp2[i][CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][0],  (Ull)kp3[i][CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][2][1],  (Ull)ip0[i][n], iofs, MSK_W1, (Ull)it[i], IMlen, 0, 0, (Ull)NULL, IMlen);\
  exe(OP_FMA, &AR[bp1][0], AR[b][0], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][1], AR[b][1], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][2], AR[b][2], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][3], AR[b][3], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn3x3x4_final(b, bp1, Force) \
  mop(OP_LDWR,   1, &BR[bp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  exe(OP_FAD, &AR[bp1][0], AR[b][0], EXP_H3210, BR[bp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][1], AR[b][1], EXP_H3210, BR[bp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][2], AR[b][2], EXP_H3210, BR[bp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][3], AR[b][3], EXP_H3210, BR[bp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  mop(OP_STWR,   1, &AR[bp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen)

//EMAX5A begin cnn3x3x4 mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
	/*2*/ for (INIT1=1,LOOP1=BATCH,img=(0-IM4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                       /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,    &img,  img,             EXP_H3210, INIT0?IM4M4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &iofs, img,             EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
                  exe(OP_ADD,    &oofs, img,             EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

                  /****in0*****/
                  mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp0[0][CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp1[0][CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp2[0][CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp3[0][CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 10KB */
                  mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)ip0[0][0],   iofs, MSK_W1, (Ull)it[0], IMlen, 0, 0, (Ull)NULL, IMlen);    /* stage#2 10KB */
                  exe(OP_FML, &AR[3][0], BR[2][2][1], EXP_H3210, BR[2][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][1], BR[2][2][1], EXP_H3210, BR[2][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][2], BR[2][2][1], EXP_H3210, BR[2][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][3], BR[2][2][1], EXP_H3210, BR[2][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		  cnn3x3x4_core1( 3, 0, 4LL, 4, 1, Force1);
		  cnn3x3x4_core1( 4, 0, 8LL, 5, 2, Force1);
		  cnn3x3x4_core1( 5, 0,12LL, 6, 3, Force1);
		  cnn3x3x4_core1( 6, 0,16LL, 7, 4, Force1);
		  cnn3x3x4_core1( 7, 0,20LL, 8, 5, Force1);
		  cnn3x3x4_core1( 8, 0,24LL, 9, 6, Force1);
		  cnn3x3x4_core1( 9, 0,28LL,10, 7, Force1);
		  cnn3x3x4_core1(10, 0,32LL,11, 8, Force1);
#if (IMAP==1)
                  /****final*****/
		  cnn3x3x4_final(11,        12, Force2);
#endif
#if (IMAP>1)
                  /****in1*****/
		  cnn3x3x4_core1(11, 1, 0LL,12, 0, Force1);
		  cnn3x3x4_core1(12, 1, 4LL,13, 1, Force1);
		  cnn3x3x4_core1(13, 1, 8LL,14, 2, Force1);
		  cnn3x3x4_core1(14, 1,12LL,15, 3, Force1);
		  cnn3x3x4_core1(15, 1,16LL,16, 4, Force1);
		  cnn3x3x4_core1(16, 1,20LL,17, 5, Force1);
		  cnn3x3x4_core1(17, 1,24LL,18, 6, Force1);
		  cnn3x3x4_core1(18, 1,28LL,19, 7, Force1);
		  cnn3x3x4_core1(19, 1,32LL,20, 8, Force1);
#endif
#if (IMAP==2)
                  /****final*****/
		  cnn3x3x4_final(20,        21, Force2);
#endif
#if (IMAP>2)
                  /****in2*****/
		  cnn3x3x4_core1(20, 2, 0LL,21, 0, Force1);
		  cnn3x3x4_core1(21, 2, 4LL,22, 1, Force1);
		  cnn3x3x4_core1(22, 2, 8LL,23, 2, Force1);
		  cnn3x3x4_core1(23, 2,12LL,24, 3, Force1);
		  cnn3x3x4_core1(24, 2,16LL,25, 4, Force1);
		  cnn3x3x4_core1(25, 2,20LL,26, 5, Force1);
		  cnn3x3x4_core1(26, 2,24LL,27, 6, Force1);
		  cnn3x3x4_core1(27, 2,28LL,28, 7, Force1);
		  cnn3x3x4_core1(28, 2,32LL,29, 8, Force1);
#endif
#if (IMAP==3)
                  /****final*****/
		  cnn3x3x4_final(29,        30, Force2);
#endif
#if (IMAP>3)
                  /****in3*****/
		  cnn3x3x4_core1(29, 3, 0LL,30, 0, Force1);
		  cnn3x3x4_core1(30, 3, 4LL,31, 1, Force1);
		  cnn3x3x4_core1(31, 3, 8LL,32, 2, Force1);
		  cnn3x3x4_core1(32, 3,12LL,33, 3, Force1);
		  cnn3x3x4_core1(33, 3,16LL,34, 4, Force1);
		  cnn3x3x4_core1(34, 3,20LL,35, 5, Force1);
		  cnn3x3x4_core1(35, 3,24LL,36, 6, Force1);
		  cnn3x3x4_core1(36, 3,28LL,37, 7, Force1);
		  cnn3x3x4_core1(37, 3,32LL,38, 8, Force1);
#endif
#if (IMAP==4)
                  /****final*****/
		  cnn3x3x4_final(38,        39, Force2);
#endif
                }
              }
            }
//EMAX5A end
            if (Force1) Force1 = 0;
            if (Force2) Force2 = 0;
printf("3");
          }
        }
      }
    }
//EMAX5A drain_dirty_lmm
    monitor_time_start(THREAD, IMAX_CPYOUT); 
    xmax_cpyout(1, out0, BATCH, OC, i_out[LANE], M, M, OC4);
    monitor_time_end(THREAD, IMAX_CPYOUT); 
}

void xmax_conv_forward_3x3x6(int THREAD, int LANE, float4D *in, float2D *kernel, float4D *out, int ksize)
{
  int   BATCH  = in->nstrides;  // 100
  int   RMGRP;
  int   IC     = in->nchannel;  // IMAP*Xin
  int   IM     = in->kstrides;  // 28
  int   OC     = out->nchannel; // W*Xout
  int   M      = out->kstrides; // 24
  int   K      = ksize;         // 5,4,3,2,1
  int   OC4    = (OC+3)&~3;
  Uint  *in0   = in->data;      // IC*IM*IM
  Uint  *ker   = kernel->data;  // OC*IC*K*K
  Uint  *out0  = out->data;     // OC*M*M
  Uint  *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *kp,  kidx, *op;
  int   pad;
  int   count, top, iset, oc, w, ic, y, x;
  Ull   IM4, M4, IM4M4, IMlen, Klen, Mlen;
  Ull   CHIP, img, rofs, cofs, iofs, oofs, k;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;
  Ull   Force1, Force2;

  if (IM == M)
    pad = 0;   /* PADÌµ¤·.in¼þ°Ï0.0¤ò²¾Äê */
  else if ((IM - K)/1 + 1 == M)
    pad = K/2; /* PADÍ­¤ê.in¼þ°ÏÆÃÊÌ°·¤¤ÉÔÍ× */
  else {
    printf("xmax_conv_forward_3x3x6 error: IM=%d K=%d M=%d\n", IM, K, M);
    printf("IM == M || (IM-K)/1+1 == M\n");
    exit(-1);
  }

  /* i_inp, i_ker, i_out¤Ï³ÎÊÝºÑ¤À¤¬À­Ç½É¾²Á¤Ë¤Ï»È¤ï¤Ê¤¤ */
  /*printf("<<<XMAX(C)>>>\n");*/
  /*printf("xmax IM=%d M=%d K=%d %d*%d*%d\n", IM, M, K, OC, BATCH*M*M, IC*K*K);*/
  /*printf("<<<XMAX(REAL)>>>\n");*/

    RMGRP = 1; /* RMGRP = 7 /*{7,16,2,7,32,1}*/
               /* RMGRP = 6 /*{7,32,2,6,32,2}*/
#undef  IMAP
#undef  W
#undef  NCHIP
/* IMAP > 1 ¡ú¡ú¡ú PBL1-1 ¡ú¡ú¡ú */
/* NCHIP  4 ¡ú¡ú¡ú PBL1-1 ¡ú¡ú¡ú */
#define IMAP  6
#define W     4
#define NCHIP 1
    /*{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,7,32,1},{7,32,2,6,32,2}*/
    /*                                    AAAAAAAAAAAAA   AAAAAAAAAAAAA */
    monitor_time_start(THREAD, IMAX_CPYIN); 
    xmax_cpyin(2, i_inp[LANE], &IM, in0, BATCH, IC, IM, M, K); /* ¤³¤Î»þÅÀ¤Ç0.0¤ÎPAD¤òÄÉ²Ã¤Ç¤­¤ë */
    xmax_cpyin(0, i_ker[LANE], &K,  ker, OC,    IC,  K, K, 1); /* ½ÐÎÏ¤Î¤ß²ó¼ý¤¹¤ì¤Ð¤è¤¤ */
    xmax_bzero(i_out[LANE], BATCH*OC4*M*M);
    monitor_time_end(THREAD, IMAX_CPYIN); 
    IM4   = IM*4;
    M4    = M*4;
    IM4M4 = IM4<<32|M4;
    IMlen = IM*BATCH*(RMGRP+(K-1));
    Klen  = OC*IC*K*K;
    Mlen  = M*BATCH*RMGRP;
    Force1 = 1;

    if (Klen > LMEM_SIZE/4/2 || IMlen > LMEM_SIZE/4/2 || Mlen > LMEM_SIZE/4/4)
      printf("   CNN3x3x6  M=%d RMGRP=%d IC=%d OC=%d outloop[M/RMGRP*IC/IMAP*OC4/NCHIP/W]=%d inloop[BATCH*M]=%d Klen=%d/%d IMlen=%d/%d Mlen=%d/%d\n",
	     (Uint)M, (Uint)RMGRP, (Uint)IC, (Uint)OC, (Uint)(M/RMGRP*IC/IMAP*OC4/NCHIP/W), (Uint)(BATCH*M), (Uint)Klen, LMEM_SIZE/4/2, (Uint)IMlen, LMEM_SIZE/4/2, (Uint)Mlen, LMEM_SIZE/4/4);

    for (top=0; top<M; top+=RMGRP) {
      for (iset=0; iset<IC; iset+=IMAP) {  /* accumulate multiple sets of IC */
	Uint *ip[IMAP], *it[IMAP], *ip0[IMAP][9];
	for (k=0; k<IMAP; k++) {
	  ip[k]     = &i_inp[LANE][(iset+k)*IM*BATCH*IM]; /* top of input#0 */
	  it[k]     = top*IM*BATCH+ip[k];
	  ip0[k][0] = ip[k]+(top+0)*IM*BATCH+0; ip0[k][1] = ip[k]+(top+0)*IM*BATCH+1; ip0[k][2] = ip[k]+(top+0)*IM*BATCH+2;
	  ip0[k][3] = ip[k]+(top+1)*IM*BATCH+0; ip0[k][4] = ip[k]+(top+1)*IM*BATCH+1; ip0[k][5] = ip[k]+(top+1)*IM*BATCH+2;
	  ip0[k][6] = ip[k]+(top+2)*IM*BATCH+0; ip0[k][7] = ip[k]+(top+2)*IM*BATCH+1; ip0[k][8] = ip[k]+(top+2)*IM*BATCH+2;
	}

        for (rofs=0; rofs<RMGRP&&(top+rofs)<M; rofs++) { /* image loop (row) */

	  for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */
	    Uint *kp0[IMAP][NCHIP],*kp1[IMAP][NCHIP],*kp2[IMAP][NCHIP],*kp3[IMAP][NCHIP];
	    Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP];
	    Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP];

            for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
              Uint choc  = CHIP*OC4/NCHIP+oc;
	      for (k=0; k<IMAP; k++) {
		kp0[k][CHIP] = (choc+0<OC) ? i_ker[LANE]+((choc+0)*IC+iset+k+0)*K*K : i_ker[LANE];
		kp1[k][CHIP] = (choc+1<OC) ? i_ker[LANE]+((choc+1)*IC+iset+k+0)*K*K : i_ker[LANE];
		kp2[k][CHIP] = (choc+2<OC) ? i_ker[LANE]+((choc+2)*IC+iset+k+0)*K*K : i_ker[LANE];
		kp3[k][CHIP] = (choc+3<OC) ? i_ker[LANE]+((choc+3)*IC+iset+k+0)*K*K : i_ker[LANE];
	      }
              op0[CHIP] = i_out[LANE]+((choc+0)*M+top)*M*BATCH; op1[CHIP] = i_out[LANE]+((choc+1)*M+top)*M*BATCH; op2[CHIP] = i_out[LANE]+((choc+2)*M+top)*M*BATCH; op3[CHIP] = i_out[LANE]+((choc+3)*M+top)*M*BATCH;
              ot0[CHIP] = i_out[LANE]+((choc+0)*M+top)*M*BATCH; ot1[CHIP] = i_out[LANE]+((choc+1)*M+top)*M*BATCH; ot2[CHIP] = i_out[LANE]+((choc+2)*M+top)*M*BATCH; ot3[CHIP] = i_out[LANE]+((choc+3)*M+top)*M*BATCH;
            }
	    Force2 = 1;

#define cnn3x3x6_core1(b, i, o, bp1, n, Force) \
  mop(OP_LDWR,   1, &BR[b][0][1],  (Ull)kp0[i][CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][0][0],  (Ull)kp1[i][CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][1],  (Ull)kp2[i][CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][0],  (Ull)kp3[i][CHIP], o, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][2][1],  (Ull)ip0[i][n], iofs, MSK_W1, (Ull)it[i], IMlen, 0, 0, (Ull)NULL, IMlen);\
  exe(OP_FMA, &AR[bp1][0], AR[b][0], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][1], AR[b][1], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][2], AR[b][2], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][3], AR[b][3], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn3x3x6_final(b, bp1, Force) \
  mop(OP_LDWR,   1, &BR[bp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  exe(OP_FAD, &AR[bp1][0], AR[b][0], EXP_H3210, BR[bp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][1], AR[b][1], EXP_H3210, BR[bp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][2], AR[b][2], EXP_H3210, BR[bp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][3], AR[b][3], EXP_H3210, BR[bp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  mop(OP_STWR,   1, &AR[bp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen)

//EMAX5A begin cnn3x3x6 mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
	/*2*/ for (INIT1=1,LOOP1=BATCH,img=(0-IM4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                       /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,    &img,  img,             EXP_H3210, INIT0?IM4M4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &iofs, img,             EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
                  exe(OP_ADD,    &oofs, img,             EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

                  /****in0*****/
                  mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp0[0][CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp1[0][CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp2[0][CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp3[0][CHIP], 0LL, MSK_D0, (Ull)i_ker[LANE], Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 10KB */
                  mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)ip0[0][0],   iofs, MSK_W1, (Ull)it[0], IMlen, 0, 0, (Ull)NULL, IMlen);    /* stage#2 10KB */
                  exe(OP_FML, &AR[3][0], BR[2][2][1], EXP_H3210, BR[2][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][1], BR[2][2][1], EXP_H3210, BR[2][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][2], BR[2][2][1], EXP_H3210, BR[2][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][3], BR[2][2][1], EXP_H3210, BR[2][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		  cnn3x3x6_core1( 3, 0, 4LL, 4, 1, Force1);
		  cnn3x3x6_core1( 4, 0, 8LL, 5, 2, Force1);
		  cnn3x3x6_core1( 5, 0,12LL, 6, 3, Force1);
		  cnn3x3x6_core1( 6, 0,16LL, 7, 4, Force1);
		  cnn3x3x6_core1( 7, 0,20LL, 8, 5, Force1);
		  cnn3x3x6_core1( 8, 0,24LL, 9, 6, Force1);
		  cnn3x3x6_core1( 9, 0,28LL,10, 7, Force1);
		  cnn3x3x6_core1(10, 0,32LL,11, 8, Force1);
#if (IMAP==1)
                  /****final*****/
		  cnn3x3x6_final(11,        12, Force2);
#endif
#if (IMAP>1)
                  /****in1*****/
		  cnn3x3x6_core1(11, 1, 0LL,12, 0, Force1);
		  cnn3x3x6_core1(12, 1, 4LL,13, 1, Force1);
		  cnn3x3x6_core1(13, 1, 8LL,14, 2, Force1);
		  cnn3x3x6_core1(14, 1,12LL,15, 3, Force1);
		  cnn3x3x6_core1(15, 1,16LL,16, 4, Force1);
		  cnn3x3x6_core1(16, 1,20LL,17, 5, Force1);
		  cnn3x3x6_core1(17, 1,24LL,18, 6, Force1);
		  cnn3x3x6_core1(18, 1,28LL,19, 7, Force1);
		  cnn3x3x6_core1(19, 1,32LL,20, 8, Force1);
#endif
#if (IMAP==2)
                  /****final*****/
		  cnn3x3x6_final(20,        21, Force2);
#endif
#if (IMAP>2)
                  /****in2*****/
		  cnn3x3x6_core1(20, 2, 0LL,21, 0, Force1);
		  cnn3x3x6_core1(21, 2, 4LL,22, 1, Force1);
		  cnn3x3x6_core1(22, 2, 8LL,23, 2, Force1);
		  cnn3x3x6_core1(23, 2,12LL,24, 3, Force1);
		  cnn3x3x6_core1(24, 2,16LL,25, 4, Force1);
		  cnn3x3x6_core1(25, 2,20LL,26, 5, Force1);
		  cnn3x3x6_core1(26, 2,24LL,27, 6, Force1);
		  cnn3x3x6_core1(27, 2,28LL,28, 7, Force1);
		  cnn3x3x6_core1(28, 2,32LL,29, 8, Force1);
#endif
#if (IMAP==3)
                  /****final*****/
		  cnn3x3x6_final(29,        30, Force2);
#endif
#if (IMAP>3)
                  /****in3*****/
		  cnn3x3x6_core1(29, 3, 0LL,30, 0, Force1);
		  cnn3x3x6_core1(30, 3, 4LL,31, 1, Force1);
		  cnn3x3x6_core1(31, 3, 8LL,32, 2, Force1);
		  cnn3x3x6_core1(32, 3,12LL,33, 3, Force1);
		  cnn3x3x6_core1(33, 3,16LL,34, 4, Force1);
		  cnn3x3x6_core1(34, 3,20LL,35, 5, Force1);
		  cnn3x3x6_core1(35, 3,24LL,36, 6, Force1);
		  cnn3x3x6_core1(36, 3,28LL,37, 7, Force1);
		  cnn3x3x6_core1(37, 3,32LL,38, 8, Force1);
#endif
#if (IMAP==4)
                  /****final*****/
		  cnn3x3x6_final(38,        39, Force2);
#endif
#if (IMAP>4)
                  /****in4*****/
		  cnn3x3x6_core1(38, 4, 0LL,39, 0, Force1);
		  cnn3x3x6_core1(39, 4, 4LL,40, 1, Force1);
		  cnn3x3x6_core1(40, 4, 8LL,41, 2, Force1);
		  cnn3x3x6_core1(41, 4,12LL,42, 3, Force1);
		  cnn3x3x6_core1(42, 4,16LL,43, 4, Force1);
		  cnn3x3x6_core1(43, 4,20LL,44, 5, Force1);
		  cnn3x3x6_core1(44, 4,24LL,45, 6, Force1);
		  cnn3x3x6_core1(45, 4,28LL,46, 7, Force1);
		  cnn3x3x6_core1(46, 4,32LL,47, 8, Force1);
#endif
#if (IMAP==5)
                  /****final*****/
		  cnn3x3x6_final(47,        48, Force2);
#endif
#if (IMAP>5)
                  /****in5*****/
		  cnn3x3x6_core1(47, 5, 0LL,48, 0, Force1);
		  cnn3x3x6_core1(48, 5, 4LL,49, 1, Force1);
		  cnn3x3x6_core1(49, 5, 8LL,50, 2, Force1);
		  cnn3x3x6_core1(50, 5,12LL,51, 3, Force1);
		  cnn3x3x6_core1(51, 5,16LL,52, 4, Force1);
		  cnn3x3x6_core1(52, 5,20LL,53, 5, Force1);
		  cnn3x3x6_core1(53, 5,24LL,54, 6, Force1);
		  cnn3x3x6_core1(54, 5,28LL,55, 7, Force1);
		  cnn3x3x6_core1(55, 5,32LL,56, 8, Force1);
#endif
#if (IMAP==6)
                  /****final*****/
		  cnn3x3x6_final(56,        57, Force2);
#endif
                }
              }
            }
//EMAX5A end
            if (Force1) Force1 = 0;
            if (Force2) Force2 = 0;
printf("3");
          }
        }
      }
    }
//EMAX5A drain_dirty_lmm
    monitor_time_start(THREAD, IMAX_CPYOUT); 
    xmax_cpyout(1, out0, BATCH, OC, i_out[LANE], M, M, OC4);
    monitor_time_end(THREAD, IMAX_CPYOUT); 
}

void xmax_conv_forward_2x2(int THREAD, int LANE, float4D *in, float2D *kernel, float4D *out, int ksize)
{
  int   BATCH  = in->nstrides;  // 100
  int   RMGRP;
  int   IC     = in->nchannel;  // IMAP*Xin
  int   IM     = in->kstrides;  // 28
  int   OC     = out->nchannel; // W*Xout
  int   M      = out->kstrides; // 24
  int   K      = ksize;         // 5,4,3,2,1
  int   OC4    = (OC+3)&~3;
  Uint  *in0   = in->data;      // IC*IM*IM
  Uint  *ker   = kernel->data;  // OC*IC*K*K
  Uint  *out0  = out->data;     // OC*M*M
  Uint  *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *kp,  kidx, *op;
  int   pad;
  int   count, top, iset, oc, w, ic, y, x;
  Ull   IM4, M4, IM4M4, IMlen, Klen, Mlen;
  Ull   CHIP, img, rofs, cofs, iofs, oofs, k;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;
  Ull   Force1, Force2;

  if (IM == M)
    pad = 0;   /* PADÌµ¤·.in¼þ°Ï0.0¤ò²¾Äê */
  else if ((IM - K)/1 + 1 == M)
    pad = K/2; /* PADÍ­¤ê.in¼þ°ÏÆÃÊÌ°·¤¤ÉÔÍ× */
  else {
    printf("xmax_conv_forward_2x2 error: IM=%d K=%d M=%d\n", IM, K, M);
    printf("IM == M || (IM-K)/1+1 == M\n");
    exit(-1);
  }

  /* i_inp, i_ker, i_out¤Ï³ÎÊÝºÑ¤À¤¬À­Ç½É¾²Á¤Ë¤Ï»È¤ï¤Ê¤¤ */
  /*printf("<<<XMAX(C)>>>\n");*/
  /*printf("xmax IM=%d M=%d K=%d %d*%d*%d\n", IM, M, K, OC, BATCH*M*M, IC*K*K);*/
  /*printf("<<<XMAX(REAL)>>>\n");*/

    RMGRP = 1; /* RMGRP = 7 /*{7,16,2,7,32,1}*/
               /* RMGRP = 6 /*{7,32,2,6,32,2}*/
#undef  IMAP
#undef  W
#undef  NCHIP
/* IMAP > 1 ¡ú¡ú¡ú PBL1-1 ¡ú¡ú¡ú */
/* NCHIP  4 ¡ú¡ú¡ú PBL1-1 ¡ú¡ú¡ú */
#define IMAP  8
#define W     4
#define NCHIP 1
    /*{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,7,32,1},{7,32,2,6,32,2}*/
    /*                                    AAAAAAAAAAAAA   AAAAAAAAAAAAA */
    monitor_time_start(THREAD, IMAX_CPYIN); 
    xmax_cpyin(2, i_inp[LANE], &IM, in0, BATCH, IC, IM, M, K); /* ¤³¤Î»þÅÀ¤Ç0.0¤ÎPAD¤òÄÉ²Ã¤Ç¤­¤ë */
    xmax_cpyin(1, i_ker[LANE], &K,  ker, OC,    IC,  K, K, 1); /* Klenºï¸º¤Î¤¿¤á¤Ë,OC¤ÈIC¤òÆþ¤ìÂØ¤¨¤ë */
    xmax_bzero(i_out[LANE], BATCH*OC4*M*M);
    monitor_time_end(THREAD, IMAX_CPYIN); 
    IM4   = IM*4;
    M4    = M*4;
    IM4M4 = IM4<<32|M4;
    IMlen = IM*BATCH*(RMGRP+(K-1));
    Klen  = IMAP*OC*K*K; /* whole of Klen > LMM_SIZE 20240101 */
    Mlen  = M*BATCH*RMGRP;
    Force1 = 1;

    if (Klen > LMEM_SIZE/4/2 || IMlen > LMEM_SIZE/4/2 || Mlen > LMEM_SIZE/4/4)
      printf("   CNN2x2  M=%d RMGRP=%d IC=%d OC=%d outloop[M/RMGRP*IC/IMAP*OC4/NCHIP/W]=%d inloop[BATCH*M]=%d Klen=%d/%d IMlen=%d/%d Mlen=%d/%d\n",
	     (Uint)M, (Uint)RMGRP, (Uint)IC, (Uint)OC, (Uint)(M/RMGRP*IC/IMAP*OC4/NCHIP/W), (Uint)(BATCH*M), (Uint)Klen, LMEM_SIZE/4/2, (Uint)IMlen, LMEM_SIZE/4/2, (Uint)Mlen, LMEM_SIZE/4/4);

    for (top=0; top<M; top+=RMGRP) {
      for (iset=0; iset<IC; iset+=IMAP) {  /* accumulate multiple sets of IC */
	Uint *kt;
	kt = i_ker[LANE]+iset*OC*K*K;
	Uint *ip[IMAP], *it[IMAP], *ip0[IMAP][4];
	for (k=0; k<IMAP; k++) {
	  ip[k]     = &i_inp[LANE][(iset+k)*IM*BATCH*IM]; /* top of input#0 */
	  it[k]     = top*IM*BATCH+ip[k];
	  ip0[k][0] = ip[k]+(top+0)*IM*BATCH+0; ip0[k][1] = ip[k]+(top+0)*IM*BATCH+1;
	  ip0[k][2] = ip[k]+(top+1)*IM*BATCH+0; ip0[k][3] = ip[k]+(top+1)*IM*BATCH+1;
	}

        for (rofs=0; rofs<RMGRP&&(top+rofs)<M; rofs++) { /* image loop (row) */

	  for (oc=0; oc<OC4/NCHIP; oc+=W) { /* set output channel */
	    Uint *kp0[IMAP][NCHIP],*kp1[IMAP][NCHIP],*kp2[IMAP][NCHIP],*kp3[IMAP][NCHIP];
	    Uint *op0[NCHIP], *op1[NCHIP], *op2[NCHIP], *op3[NCHIP];
	    Uint *ot0[NCHIP], *ot1[NCHIP], *ot2[NCHIP], *ot3[NCHIP];

            for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
              Uint choc  = CHIP*OC4/NCHIP+oc;
	      for (k=0; k<IMAP; k++) {
		kp0[k][CHIP] = (choc+0<OC) ? i_ker[LANE]+((iset+k)*OC+choc+0)*K*K : i_ker[LANE]; /* +iset+k is changed to +k for reduce Klen. 20240101 */
		kp1[k][CHIP] = (choc+1<OC) ? i_ker[LANE]+((iset+k)*OC+choc+1)*K*K : i_ker[LANE];
		kp2[k][CHIP] = (choc+2<OC) ? i_ker[LANE]+((iset+k)*OC+choc+2)*K*K : i_ker[LANE];
		kp3[k][CHIP] = (choc+3<OC) ? i_ker[LANE]+((iset+k)*OC+choc+3)*K*K : i_ker[LANE];
	      }
              op0[CHIP] = i_out[LANE]+((choc+0)*M+top)*M*BATCH; op1[CHIP] = i_out[LANE]+((choc+1)*M+top)*M*BATCH; op2[CHIP] = i_out[LANE]+((choc+2)*M+top)*M*BATCH; op3[CHIP] = i_out[LANE]+((choc+3)*M+top)*M*BATCH;
              ot0[CHIP] = i_out[LANE]+((choc+0)*M+top)*M*BATCH; ot1[CHIP] = i_out[LANE]+((choc+1)*M+top)*M*BATCH; ot2[CHIP] = i_out[LANE]+((choc+2)*M+top)*M*BATCH; ot3[CHIP] = i_out[LANE]+((choc+3)*M+top)*M*BATCH;
            }
	    Force2 = 1;

#define cnn2x2_core1(b, i, o, bp1, n, Force) \
  mop(OP_LDWR,   1, &BR[b][0][1],  (Ull)kp0[i][CHIP], o, MSK_D0, (Ull)kt, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][0][0],  (Ull)kp1[i][CHIP], o, MSK_D0, (Ull)kt, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][1],  (Ull)kp2[i][CHIP], o, MSK_D0, (Ull)kt, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][1][0],  (Ull)kp3[i][CHIP], o, MSK_D0, (Ull)kt, Klen, 0, Force, (Ull)NULL, Klen);\
  mop(OP_LDWR,   1, &BR[b][2][1],  (Ull)ip0[i][n], iofs, MSK_W1, (Ull)it[i], IMlen, 0, 0, (Ull)NULL, IMlen);\
  exe(OP_FMA, &AR[bp1][0], AR[b][0], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][1], AR[b][1], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][2], AR[b][2], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FMA, &AR[bp1][3], AR[b][3], EXP_H3210, BR[b][2][1], EXP_H3210, BR[b][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define cnn2x2_final(b, bp1, Force) \
  mop(OP_LDWR,   1, &BR[bp1][0][1],  (Ull)op0[CHIP], oofs, MSK_W0, (Ull)ot0[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][1][1],  (Ull)op1[CHIP], oofs, MSK_W0, (Ull)ot1[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][2][1],  (Ull)op2[CHIP], oofs, MSK_W0, (Ull)ot2[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_LDWR,   1, &BR[bp1][3][1],  (Ull)op3[CHIP], oofs, MSK_W0, (Ull)ot3[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  exe(OP_FAD, &AR[bp1][0], AR[b][0], EXP_H3210, BR[bp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][1], AR[b][1], EXP_H3210, BR[bp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][2], AR[b][2], EXP_H3210, BR[bp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  exe(OP_FAD, &AR[bp1][3], AR[b][3], EXP_H3210, BR[bp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
  mop(OP_STWR,   1, &AR[bp1][0], oofs, (Ull)op0[CHIP], MSK_D0, (Ull)ot0[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][1], oofs, (Ull)op1[CHIP], MSK_D0, (Ull)ot1[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][2], oofs, (Ull)op2[CHIP], MSK_D0, (Ull)ot2[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen);\
  mop(OP_STWR,   1, &AR[bp1][3], oofs, (Ull)op3[CHIP], MSK_D0, (Ull)ot3[CHIP], Mlen, 0, Force, (Ull)NULL, Mlen)

//EMAX5A begin cnn2x2 mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
	/*2*/ for (INIT1=1,LOOP1=BATCH,img=(0-IM4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                       /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,    &img,  img,             EXP_H3210, INIT0?IM4M4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, 4LL<<32|4LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                  exe(OP_ADD,    &iofs, img,             EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
                  exe(OP_ADD,    &oofs, img,             EXP_H3210, cofs,          EXP_H3210, 0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

                  /****in0*****/
                  mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp0[0][CHIP], 0LL, MSK_D0, (Ull)kt, Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp1[0][CHIP], 0LL, MSK_D0, (Ull)kt, Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp2[0][CHIP], 0LL, MSK_D0, (Ull)kt, Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 */
                  mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp3[0][CHIP], 0LL, MSK_D0, (Ull)kt, Klen, 0, Force1, (Ull)NULL, Klen); /* stage#2 10KB */
                  mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)ip0[0][0],   iofs, MSK_W1, (Ull)it[0], IMlen, 0, 0, (Ull)NULL, IMlen);    /* stage#2 10KB */
                  exe(OP_FML, &AR[3][0], BR[2][2][1], EXP_H3210, BR[2][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][1], BR[2][2][1], EXP_H3210, BR[2][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][2], BR[2][2][1], EXP_H3210, BR[2][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                  exe(OP_FML, &AR[3][3], BR[2][2][1], EXP_H3210, BR[2][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
		  cnn2x2_core1( 3, 0, 4LL, 4, 1, Force1);
		  cnn2x2_core1( 4, 0, 8LL, 5, 2, Force1);
		  cnn2x2_core1( 5, 0,12LL, 6, 3, Force1);
                  /****in1*****/
		  cnn2x2_core1( 6, 1, 0LL, 7, 0, Force1);
		  cnn2x2_core1( 7, 1, 4LL, 8, 1, Force1);
		  cnn2x2_core1( 8, 1, 8LL, 9, 2, Force1);
		  cnn2x2_core1( 9, 1,12LL,10, 3, Force1);
                  /****in2*****/
		  cnn2x2_core1(10, 2, 0LL,11, 0, Force1);
		  cnn2x2_core1(11, 2, 4LL,12, 1, Force1);
		  cnn2x2_core1(12, 2, 8LL,13, 2, Force1);
		  cnn2x2_core1(13, 2,12LL,14, 3, Force1);
                  /****in3*****/
		  cnn2x2_core1(14, 3, 0LL,15, 0, Force1);
		  cnn2x2_core1(15, 3, 4LL,16, 1, Force1);
		  cnn2x2_core1(16, 3, 8LL,17, 2, Force1);
		  cnn2x2_core1(17, 3,12LL,18, 3, Force1);
#if (IMAP==4)
		  /****final*****/
		  cnn2x2_final(18,        19, Force2);
#endif
#if (IMAP>4)
                  /****in4*****/
		  cnn2x2_core1(18, 4, 0LL,19, 0, Force1);
		  cnn2x2_core1(19, 4, 4LL,20, 1, Force1);
		  cnn2x2_core1(20, 4, 8LL,21, 2, Force1);
		  cnn2x2_core1(21, 4,12LL,22, 3, Force1);
                  /****in5*****/
		  cnn2x2_core1(22, 5, 0LL,23, 0, Force1);
		  cnn2x2_core1(23, 5, 4LL,24, 1, Force1);
		  cnn2x2_core1(24, 5, 8LL,25, 2, Force1);
		  cnn2x2_core1(25, 5,12LL,26, 3, Force1);
                  /****in6*****/
		  cnn2x2_core1(26, 6, 0LL,27, 0, Force1);
		  cnn2x2_core1(27, 6, 4LL,28, 1, Force1);
		  cnn2x2_core1(28, 6, 8LL,29, 2, Force1);
		  cnn2x2_core1(29, 6,12LL,30, 3, Force1);
                  /****in7*****/
		  cnn2x2_core1(30, 7, 0LL,31, 0, Force1);
		  cnn2x2_core1(31, 7, 4LL,32, 1, Force1);
		  cnn2x2_core1(32, 7, 8LL,33, 2, Force1);
		  cnn2x2_core1(33, 7,12LL,34, 3, Force1);
#endif
#if (IMAP==8)
		  /****final*****/
		  cnn2x2_final(34,        35, Force2);
#endif
                }
              }
            }
//EMAX5A end
            if (Force1) Force1 = 0;
            if (Force2) Force2 = 0;
switch (M) {
case 7:  printf("w");break;
case 6:  printf("x");break;
case 2:  printf("y");break;
default: printf("z");break;
    }
          }
        }
      }
    }
//EMAX5A drain_dirty_lmm
    monitor_time_start(THREAD, IMAX_CPYOUT); 
    xmax_cpyout(1, out0, BATCH, OC, i_out[LANE], M, M, OC4);
    monitor_time_end(THREAD, IMAX_CPYOUT); 
}

void xmax_sgemm00_48(int, int, int, int, int, float*, float*, float*); /* C=A*B */
void xmax_sgemm00_32(int, int, int, int, int, float*, float*, float*); /* C=A*B */
void xmax_sgemm00_40(int, int, int, int, int, float*, float*, float*); /* C=A*B */

void xmax_sgemm00(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  if (ka % 48 == 0)
    xmax_sgemm00_48(THREAD, LANE, m, n, ka, A, B, C); /* C=A*B */
  else if (ka % 32 == 0)
    xmax_sgemm00_32(THREAD, LANE, m, n, ka, A, B, C); /* C=A*B */
  else if (ka % 40 == 0)
    xmax_sgemm00_40(THREAD, LANE, m, n, ka, A, B, C); /* C=A*B */
  else {
    printf("xmax_sgemm00 error: ka=%d\n", ka);
    exit(-1);
  }
}

void xmax_sgemm00_48(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  /*  ¨£¨¡¨¡¨¡¨¡¨¡¨¤convolution¤Î¾ì¹ç                                                  */
  /*  ¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤B¤¬Ê£¿ô¤È¹Í¤¨¤ë                                                  */
  /*  ¨¢¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤        ¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤                       */
  /*  ¨¢¨¢¨¢b         ¨¢¨¢a a a a a ¨¢¨¢RMGRP   ¨¢o o o o o ¨¢¨¢RMGRP                  */
  /*  ¨¢¨¢¨¢b         ¨©¨¢          ¨¢¨©/CHIP   ¨¢          ¨¢¨©/CHIP                  */
  /*  ¨¢¨¢¨¢b   B0   b¨¢¨¢ A(weight)¨¢¨¢        ¨¢   out    ¨¢¨¢ mm¤Î¾ì¹ç¤Ï¹Ô¤ÇÊ¬³ä    */
  /*  ¨¦¨¢¨¢b        l¨©¨¢          ¨¢¨©        ¨¢          ¨¢¨© cnn¤Î¾ì¹ç¤Ïout¤ÇÊ¬³ä  */
  /*    ¨¦¨¢b        k¨¢¨¢blk       ¨¢¨¢        ¨¢blk       ¨¢¨¢                       */
  /*      ¨¦¨¡¨¡¨¡¨¡¨¡¨¥¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥        ¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥                       */

  int  RMGRP, Alen, Blen, Clen;
  int  row, col, k;
  int  count, top, blk;
  Ull  KA4, N, n4, KA4n4;
  Ull  CHIP, rofs, cofs, oofs;
  Ull  cofslimit1, cofslimit2, cofslimit3;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1, ex2, ex3;
  Ull  Force;

#undef  IMAP
#undef  W
#undef  H
#undef  NCHIP
#define IMAP  1
#define W     4LL
#define H     48
/* NCHIP  4 ¡ú¡ú¡ú nakashima ¡ú¡ú¡ú */
#define NCHIP 1
  N = (n+3)&~3;
  monitor_time_start(THREAD, IMAX_CPYIN);
  xmax_cpyin(3, i_m0A[LANE], &m, A, 1, 1, m, ka, 1);
  xmax_cpyin(3, i_m0B[LANE], &n, B, 1, 1, n, ka, 1);
  xmax_bzero(i_m0C[LANE], m*n); /* m*N */
  monitor_time_end(THREAD, IMAX_CPYIN);
  /*  m=100/NCHIP(4)¤ò³ä¤êÀÚ¤ì¤ëÃÍ¤È¤·¤Æ,RMGRP=5              */
  /* xsim/xsim-zynq.emax7+dma -x -t -I1 -C4 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ka=288,288*RMGRP*4=5KB(<64KB)¤È¤Ê¤êLMM¤ËÆþ¤ë            */
  /* xsim/xsim-zynq.emax7+dma -x -t -I0 -C1 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ich=9, ka=1296,1296*RMGRP(5)*4=26KB(<64KB)¤È¤Ê¤êrsim¤ÏLMM¤ËÆþ¤ë */
  /*  ich=17,ka=2448,2448*RMGRP(5)*4=49KB(<64KB)¤È¤Ê¤êssim¤ÏLMM¤ËÆþ¤ë */
  RMGRP = (LMEM_SIZE/4/2)/ka>100 ? 100:
          (LMEM_SIZE/4/2)/ka>20  ? 20:
          (LMEM_SIZE/4/2)/ka>10  ? 10:
          (LMEM_SIZE/4/2)/ka>5   ? 5:2;           /* CIFAR10:6KB,MNIST:50KB */
  Alen  = ka*RMGRP;      /* 288*5*4B  = 5760B    */
  Blen  = n;             /* 10/2      = 5        */
  Clen  = n*RMGRP;       /* 10*5*4B   = 200B     */
  KA4   = ka*4;          /* 288*4B               */
  n4    = n*4;           /* 10*4B                */
  KA4n4 = KA4<<32|n4;

  if (Blen > LMEM_SIZE/4/2 || Alen > LMEM_SIZE/4/2 || Clen > LMEM_SIZE/4)
    printf("   GEMM00  m=%d n=%d ka=%d(/H) outloop[m/NCHIP/RMGRP*ka/H]=%d inloop[RMGRP*N/W]=%d Blen=%d/%d Alen=%d/%d Clen=%d/%d\n",
	   (Uint)m, (Uint)n, (Uint)ka, (Uint)(m/NCHIP/RMGRP*ka/H), (Uint)(RMGRP*N/W), (Uint)Blen, LMEM_SIZE/4/2, (Uint)Alen, LMEM_SIZE/4/2, (Uint)Clen, LMEM_SIZE/4);

  for (top=0; top<m/NCHIP; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    Force = 1;
    for (blk=0; blk<ka; blk+=H) { /* 3½Å¥ë¡¼¥×Å¸³«¤Î³°Â¦ÂÐ¾Ý */
      typedef struct {Uint i[4]} Ui4;
      Uint *a0[NCHIP];
      Uint *a[H][NCHIP];
      Ui4  *b[H], *b0[H], *b1[H], *b2[H], *b3[H];
      Ui4  *c0[NCHIP];
      Ui4  *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
      for (k=0; k<H; k++) {
	b[k] = i_m0B[LANE]+(blk+k)*n; b0[k] = b[k]; b1[k] = (Uint*)b[k]+1; b2[k] = (Uint*)b[k]+2;  b3[k] = (Uint*)b[k]+3; 
      }
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
	a0[CHIP] = i_m0A[LANE]+(CHIP*m/NCHIP+top)*ka;
	for (k=0; k<H; k++)
	  a[k][CHIP] = a0[CHIP]+blk+k;
	c0[CHIP] = i_m0C[LANE]+(CHIP*m/NCHIP+top)*n;
	c00[CHIP]= (Uint*)c0[CHIP]+0; c01[CHIP]= (Uint*)c0[CHIP]+1; c02[CHIP]= (Uint*)c0[CHIP]+2; c03[CHIP]= (Uint*)c0[CHIP]+3;
      }
      cofslimit1 = n4- 4; /* cofs32 < 36 x */
      cofslimit2 = n4- 8; /* cofs32 < 32 x */
      cofslimit3 = n4-12; /* cofs32 < 28 x */

#define sgemm00_48_core1(r, rm1, rp1) \
	    mop(OP_LDWR,   1, &BR[r][0][1],  (Ull)b0[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][0][0],  (Ull)b1[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][1][1],  (Ull)b2[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][1][0],  (Ull)b3[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][2][1],  (Ull)a[rm1][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);\
	    exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define sgemm00_48_final(r, rp1, Force) \
	    exe(OP_CMP_LT,   &cc1, cofs, EXP_H3210, cofslimit1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_CMP_LT,   &cc2, cofs, EXP_H3210, cofslimit2, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_CMP_LT,   &cc3, cofs, EXP_H3210, cofslimit3, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    mop(OP_LDWR,   1, &BR[rp1][0][1],  (Ull)c00[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][1][1],  (Ull)c01[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][2][1],  (Ull)c02[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][3][1],  (Ull)c03[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    mop(OP_STWR,   1, &AR[rp1][0],     (Ull)oofs, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex1,   0, 0, 0, cc1, 0xaaaa);\
	    mop(OP_STWR, ex1, &AR[rp1][1],     (Ull)oofs, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex2,   0, 0, 0, cc2, 0xaaaa);\
	    mop(OP_STWR, ex2, &AR[rp1][2],     (Ull)oofs, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex3,   0, 0, 0, cc3, 0xaaaa);\
	    mop(OP_STWR, ex3, &AR[rp1][3],     (Ull)oofs, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen)

//EMAX5A begin sgemm00_48 mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-KA4)<<32|((0-n4)&0xffffffff); LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=N/W,cofs=(0-W*4)<<32|((0-W*4)&0xffffffff); LOOP0--; INIT0=0) {  /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
	    exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (W*4)<<32|(W*4), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);/* stage#0 */
	    exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?KA4n4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* stage#0 */
	    exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffff, OP_NOP, 0LL);           /* stage#1 */

	    mop(OP_LDWR,   1, &BR[1][0][1],  (Ull)b0[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][0][0],  (Ull)b1[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][1][1],  (Ull)b2[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][1][0],  (Ull)b3[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 2KB */
	    mop(OP_LDWR,   1, &BR[1][2][1],  (Ull)a[0][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);/* stage#1 16KB */
	    exe(OP_FML, &AR[2][0], BR[1][0][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][1], BR[1][0][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][2], BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][3], BR[1][1][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	      
	    sgemm00_48_core1( 2,  1,  3);
	    sgemm00_48_core1( 3,  2,  4);
	    sgemm00_48_core1( 4,  3,  5);
	    sgemm00_48_core1( 5,  4,  6);
	    sgemm00_48_core1( 6,  5,  7);
	    sgemm00_48_core1( 7,  6,  8);
	    sgemm00_48_core1( 8,  7,  9);
	    sgemm00_48_core1( 9,  8, 10);
	    sgemm00_48_core1(10,  9, 11);
	    sgemm00_48_core1(11, 10, 12);
	    sgemm00_48_core1(12, 11, 13);
	    sgemm00_48_core1(13, 12, 14);
	    sgemm00_48_core1(14, 13, 15);
	    sgemm00_48_core1(15, 14, 16);
	    sgemm00_48_core1(16, 15, 17);
	    sgemm00_48_core1(17, 16, 18);
	    sgemm00_48_core1(18, 17, 19);
	    sgemm00_48_core1(19, 18, 20);
	    sgemm00_48_core1(20, 19, 21);
	    sgemm00_48_core1(21, 20, 22);
	    sgemm00_48_core1(22, 21, 23);
	    sgemm00_48_core1(23, 22, 24);
	    sgemm00_48_core1(24, 23, 25);
#if (H==24)
	    sgemm00_48_final(25,     27, Force);
#endif
#if (H>24)
	    sgemm00_48_core1(25, 24, 26);
	    sgemm00_48_core1(26, 25, 27);
	    sgemm00_48_core1(27, 26, 28);
	    sgemm00_48_core1(28, 27, 29);
	    sgemm00_48_core1(29, 28, 30);
	    sgemm00_48_core1(30, 29, 31);
	    sgemm00_48_core1(31, 30, 32);
	    sgemm00_48_core1(32, 31, 33);
	    sgemm00_48_core1(33, 32, 34);
	    sgemm00_48_core1(34, 33, 35);
	    sgemm00_48_core1(35, 34, 36);
	    sgemm00_48_core1(36, 35, 37);
	    sgemm00_48_core1(37, 36, 38);
	    sgemm00_48_core1(38, 37, 39);
	    sgemm00_48_core1(39, 38, 40);
	    sgemm00_48_core1(40, 39, 41);
	    sgemm00_48_core1(41, 40, 42);
	    sgemm00_48_core1(42, 41, 43);
	    sgemm00_48_core1(43, 42, 44);
	    sgemm00_48_core1(44, 43, 45);
	    sgemm00_48_core1(45, 44, 46);
	    sgemm00_48_core1(46, 45, 47);
	    sgemm00_48_core1(47, 46, 48);
	    sgemm00_48_core1(48, 47, 49); /* 288/6 H=48 */
#endif
#if (H==48)
	    /****final*****/
	    sgemm00_48_final(49,     51, Force);
#endif
          }
        }
      }
//EMAX5A end
      if (Force) Force = 0; /* reset wdat load to LMM */
printf("*");
    }
  }
//EMAX5A drain_dirty_lmm
  monitor_time_start(THREAD, IMAX_CPYOUT);
  xmax_cpyout(2, C, 1, 1, i_m0C[LANE], m, n, n); /* i_m0C is contiguous w/ CEX+ST */
  monitor_time_end(THREAD, IMAX_CPYOUT);
}

void xmax_sgemm00_32(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  /*  ¨£¨¡¨¡¨¡¨¡¨¡¨¤convolution¤Î¾ì¹ç                                                  */
  /*  ¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤B¤¬Ê£¿ô¤È¹Í¤¨¤ë                                                  */
  /*  ¨¢¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤        ¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤                       */
  /*  ¨¢¨¢¨¢b         ¨¢¨¢a a a a a ¨¢¨¢RMGRP   ¨¢o o o o o ¨¢¨¢RMGRP                  */
  /*  ¨¢¨¢¨¢b         ¨©¨¢          ¨¢¨©/CHIP   ¨¢          ¨¢¨©/CHIP                  */
  /*  ¨¢¨¢¨¢b   B0   b¨¢¨¢ A(weight)¨¢¨¢        ¨¢   out    ¨¢¨¢ mm¤Î¾ì¹ç¤Ï¹Ô¤ÇÊ¬³ä    */
  /*  ¨¦¨¢¨¢b        l¨©¨¢          ¨¢¨©        ¨¢          ¨¢¨© cnn¤Î¾ì¹ç¤Ïout¤ÇÊ¬³ä  */
  /*    ¨¦¨¢b        k¨¢¨¢blk       ¨¢¨¢        ¨¢blk       ¨¢¨¢                       */
  /*      ¨¦¨¡¨¡¨¡¨¡¨¡¨¥¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥        ¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥                       */

  int  RMGRP, Alen, Blen, Clen;
  int  row, col, k;
  int  count, top, blk;
  Ull  KA4, N, n4, KA4n4;
  Ull  CHIP, rofs, cofs, oofs;
  Ull  cofslimit1, cofslimit2, cofslimit3;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1, ex2, ex3;
  Ull  Force;

#undef  IMAP
#undef  W
#undef  H
#undef  NCHIP
#define IMAP  1
#define W     4LL
#define H     32
/* NCHIP  4 ¡ú¡ú¡ú nakashima ¡ú¡ú¡ú */
#define NCHIP 1
  N = (n+3)&~3;
  monitor_time_start(THREAD, IMAX_CPYIN);
  xmax_cpyin(3, i_m0A[LANE], &m, A, 1, 1, m, ka, 1);
  xmax_cpyin(3, i_m0B[LANE], &n, B, 1, 1, n, ka, 1);
  xmax_bzero(i_m0C[LANE], m*n); /* m*N */
  monitor_time_end(THREAD, IMAX_CPYIN);
  /*  m=100/NCHIP(4)¤ò³ä¤êÀÚ¤ì¤ëÃÍ¤È¤·¤Æ,RMGRP=5              */
  /* xsim/xsim-zynq.emax7+dma -x -t -I1 -C4 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ka=288,288*RMGRP*4=5KB(<64KB)¤È¤Ê¤êLMM¤ËÆþ¤ë            */
  /* xsim/xsim-zynq.emax7+dma -x -t -I0 -C1 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ich=9, ka=1296,1296*RMGRP(5)*4=26KB(<64KB)¤È¤Ê¤êrsim¤ÏLMM¤ËÆþ¤ë */
  /*  ich=17,ka=2448,2448*RMGRP(5)*4=49KB(<64KB)¤È¤Ê¤êssim¤ÏLMM¤ËÆþ¤ë */
  RMGRP = (LMEM_SIZE/4/2)/ka>100 ? 100:
          (LMEM_SIZE/4/2)/ka>20  ? 20:
          (LMEM_SIZE/4/2)/ka>10  ? 10:
          (LMEM_SIZE/4/2)/ka>5   ? 5:2;           /* CIFAR10:6KB,MNIST:50KB */
  Alen  = ka*RMGRP;      /* 288*5*4B  = 5760B    */
  Blen  = n;             /* 10/2      = 5        */
  Clen  = n*RMGRP;       /* 10*5*4B   = 200B     */
  KA4   = ka*4;          /* 288*4B               */
  n4    = n*4;           /* 10*4B                */
  KA4n4 = KA4<<32|n4;

  if (Blen > LMEM_SIZE/4/2 || Alen > LMEM_SIZE/4/2 || Clen > LMEM_SIZE/4)
    printf("   GEMM00  m=%d n=%d ka=%d(/H) outloop[m/NCHIP/RMGRP*ka/H]=%d inloop[RMGRP*N/W]=%d Blen=%d/%d Alen=%d/%d Clen=%d/%d\n",
	   (Uint)m, (Uint)n, (Uint)ka, (Uint)(m/NCHIP/RMGRP*ka/H), (Uint)(RMGRP*N/W), (Uint)Blen, LMEM_SIZE/4/2, (Uint)Alen, LMEM_SIZE/4/2, (Uint)Clen, LMEM_SIZE/4);

  for (top=0; top<m/NCHIP; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    Force = 1;
    for (blk=0; blk<ka; blk+=H) { /* 3½Å¥ë¡¼¥×Å¸³«¤Î³°Â¦ÂÐ¾Ý */
      typedef struct {Uint i[4]} Ui4;
      Uint *a0[NCHIP];
      Uint *a[H][NCHIP];
      Ui4  *b[H], *b0[H], *b1[H], *b2[H], *b3[H];
      Ui4  *c0[NCHIP];
      Ui4  *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
      for (k=0; k<H; k++) {
	b[k] = i_m0B[LANE]+(blk+k)*n; b0[k] = b[k]; b1[k] = (Uint*)b[k]+1; b2[k] = (Uint*)b[k]+2;  b3[k] = (Uint*)b[k]+3; 
      }
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
	a0[CHIP] = i_m0A[LANE]+(CHIP*m/NCHIP+top)*ka;
	for (k=0; k<H; k++)
	  a[k][CHIP] = a0[CHIP]+blk+k;
	c0[CHIP] = i_m0C[LANE]+(CHIP*m/NCHIP+top)*n;
	c00[CHIP]= (Uint*)c0[CHIP]+0; c01[CHIP]= (Uint*)c0[CHIP]+1; c02[CHIP]= (Uint*)c0[CHIP]+2; c03[CHIP]= (Uint*)c0[CHIP]+3;
      }
      cofslimit1 = n4- 4; /* cofs32 < 36 x */
      cofslimit2 = n4- 8; /* cofs32 < 32 x */
      cofslimit3 = n4-12; /* cofs32 < 28 x */

#define sgemm00_32_core1(r, rm1, rp1) \
	    mop(OP_LDWR,   1, &BR[r][0][1],  (Ull)b0[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][0][0],  (Ull)b1[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][1][1],  (Ull)b2[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][1][0],  (Ull)b3[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][2][1],  (Ull)a[rm1][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);\
	    exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define sgemm00_32_final(r, rp1, Force) \
	    exe(OP_CMP_LT,   &cc1, cofs, EXP_H3210, cofslimit1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_CMP_LT,   &cc2, cofs, EXP_H3210, cofslimit2, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_CMP_LT,   &cc3, cofs, EXP_H3210, cofslimit3, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    mop(OP_LDWR,   1, &BR[rp1][0][1],  (Ull)c00[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][1][1],  (Ull)c01[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][2][1],  (Ull)c02[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][3][1],  (Ull)c03[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    mop(OP_STWR,   1, &AR[rp1][0],     (Ull)oofs, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex1,   0, 0, 0, cc1, 0xaaaa);\
	    mop(OP_STWR, ex1, &AR[rp1][1],     (Ull)oofs, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex2,   0, 0, 0, cc2, 0xaaaa);\
	    mop(OP_STWR, ex2, &AR[rp1][2],     (Ull)oofs, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex3,   0, 0, 0, cc3, 0xaaaa);\
	    mop(OP_STWR, ex3, &AR[rp1][3],     (Ull)oofs, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen)

//EMAX5A begin sgemm00_32 mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-KA4)<<32|((0-n4)&0xffffffff); LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=N/W,cofs=(0-W*4)<<32|((0-W*4)&0xffffffff); LOOP0--; INIT0=0) {  /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
            exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (W*4)<<32|(W*4), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);/* stage#0 */
	    exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?KA4n4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* stage#0 */
	    exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffff, OP_NOP, 0LL);           /* stage#1 */

	    mop(OP_LDWR,   1, &BR[1][0][1],  (Ull)b0[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][0][0],  (Ull)b1[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][1][1],  (Ull)b2[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][1][0],  (Ull)b3[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 2KB */
	    mop(OP_LDWR,   1, &BR[1][2][1],  (Ull)a[0][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);/* stage#1 16KB */
	    exe(OP_FML, &AR[2][0], BR[1][0][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][1], BR[1][0][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][2], BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][3], BR[1][1][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */

	    sgemm00_32_core1( 2,  1,  3);
	    sgemm00_32_core1( 3,  2,  4);
	    sgemm00_32_core1( 4,  3,  5);
	    sgemm00_32_core1( 5,  4,  6);
	    sgemm00_32_core1( 6,  5,  7);
	    sgemm00_32_core1( 7,  6,  8);
	    sgemm00_32_core1( 8,  7,  9);
	    sgemm00_32_core1( 9,  8, 10);
	    sgemm00_32_core1(10,  9, 11);
	    sgemm00_32_core1(11, 10, 12);
	    sgemm00_32_core1(12, 11, 13);
	    sgemm00_32_core1(13, 12, 14);
	    sgemm00_32_core1(14, 13, 15);
	    sgemm00_32_core1(15, 14, 16);
	    sgemm00_32_core1(16, 15, 17);
#if (H==16)
	    sgemm00_32_final(17,     19, Force);
#endif
#if (H>16)
	    sgemm00_32_core1(17, 16, 18);
	    sgemm00_32_core1(18, 17, 19);
	    sgemm00_32_core1(19, 18, 20);
	    sgemm00_32_core1(20, 19, 21);
	    sgemm00_32_core1(21, 20, 22);
	    sgemm00_32_core1(22, 21, 23);
	    sgemm00_32_core1(23, 22, 24);
	    sgemm00_32_core1(24, 23, 25);
	    sgemm00_32_core1(25, 24, 26);
	    sgemm00_32_core1(26, 25, 27);
	    sgemm00_32_core1(27, 26, 28);
	    sgemm00_32_core1(28, 27, 29);
	    sgemm00_32_core1(29, 28, 30);
	    sgemm00_32_core1(30, 29, 31);
	    sgemm00_32_core1(31, 30, 32);
	    sgemm00_32_core1(32, 31, 33);
#endif
#if (H==32)
	    /****final*****/
	    sgemm00_32_final(33,     35, Force);
#endif
          }
        }
      }
//EMAX5A end
      if (Force) Force = 0; /* reset wdat load to LMM */
printf("*");
    }
  }
//EMAX5A drain_dirty_lmm
  monitor_time_start(THREAD, IMAX_CPYOUT);
  xmax_cpyout(2, C, 1, 1, i_m0C[LANE], m, n, n); /* i_m0C is contiguous w/ CEX+ST */
  monitor_time_end(THREAD, IMAX_CPYOUT);
}

void xmax_sgemm00_40(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  /*  ¨£¨¡¨¡¨¡¨¡¨¡¨¤convolution¤Î¾ì¹ç                                                  */
  /*  ¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤B¤¬Ê£¿ô¤È¹Í¤¨¤ë                                                  */
  /*  ¨¢¨¢¨£¨¡¨¡¨¡¨¡¨ª¨¤¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤        ¨£¨¡¨¡¨¡¨¡¨¡¨¤¨¤                       */
  /*  ¨¢¨¢¨¢b         ¨¢¨¢a a a a a ¨¢¨¢RMGRP   ¨¢o o o o o ¨¢¨¢RMGRP                  */
  /*  ¨¢¨¢¨¢b         ¨©¨¢          ¨¢¨©/CHIP   ¨¢          ¨¢¨©/CHIP                  */
  /*  ¨¢¨¢¨¢b   B0   b¨¢¨¢ A(weight)¨¢¨¢        ¨¢   out    ¨¢¨¢ mm¤Î¾ì¹ç¤Ï¹Ô¤ÇÊ¬³ä    */
  /*  ¨¦¨¢¨¢b        l¨©¨¢          ¨¢¨©        ¨¢          ¨¢¨© cnn¤Î¾ì¹ç¤Ïout¤ÇÊ¬³ä  */
  /*    ¨¦¨¢b        k¨¢¨¢blk       ¨¢¨¢        ¨¢blk       ¨¢¨¢                       */
  /*      ¨¦¨¡¨¡¨¡¨¡¨¡¨¥¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥        ¨¦¨¡¨ª¨¡¨ª¨¡¨¥¨¥                       */

  int  RMGRP, Alen, Blen, Clen;
  int  row, col, k;
  int  count, top, blk;
  Ull  KA4, N, n4, KA4n4;
  Ull  CHIP, rofs, cofs, oofs;
  Ull  cofslimit1, cofslimit2, cofslimit3;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1, ex2, ex3;
  Ull  Force;

#undef  IMAP
#undef  W
#undef  H
#undef  NCHIP
#define IMAP  1
#define W     4LL
#define H     40
/* NCHIP  4 ¡ú¡ú¡ú nakashima ¡ú¡ú¡ú */
#define NCHIP 1
  N = (n+3)&~3;
  monitor_time_start(THREAD, IMAX_CPYIN);
  xmax_cpyin(3, i_m0A[LANE], &m, A, 1, 1, m, ka, 1);
  xmax_cpyin(3, i_m0B[LANE], &n, B, 1, 1, n, ka, 1);
  xmax_bzero(i_m0C[LANE], m*n); /* m*N */
  monitor_time_end(THREAD, IMAX_CPYIN);
  /*  m=100/NCHIP(4)¤ò³ä¤êÀÚ¤ì¤ëÃÍ¤È¤·¤Æ,RMGRP=5              */
  /* xsim/xsim-zynq.emax7+dma -x -t -I1 -C4 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ka=288,288*RMGRP*4=5KB(<64KB)¤È¤Ê¤êLMM¤ËÆþ¤ë            */
  /* xsim/xsim-zynq.emax7+dma -x -t -I0 -C1 -F1¤Î¾ì¹ç RMGRP=5 */
  /*  ich=9, ka=1296,1296*RMGRP(5)*4=26KB(<64KB)¤È¤Ê¤êrsim¤ÏLMM¤ËÆþ¤ë */
  /*  ich=17,ka=2448,2448*RMGRP(5)*4=49KB(<64KB)¤È¤Ê¤êssim¤ÏLMM¤ËÆþ¤ë */
  RMGRP = (LMEM_SIZE/4/2)/ka>100 ? 100:
          (LMEM_SIZE/4/2)/ka>20  ? 20:
          (LMEM_SIZE/4/2)/ka>10  ? 10:
          (LMEM_SIZE/4/2)/ka>5   ? 5:2;           /* CIFAR10:6KB,MNIST:50KB */
  Alen  = ka*RMGRP;      /* 288*5*4B  = 5760B    */
  Blen  = n;             /* 10/2      = 5        */
  Clen  = n*RMGRP;       /* 10*5*4B   = 200B     */
  KA4   = ka*4;          /* 288*4B               */
  n4    = n*4;           /* 10*4B                */
  KA4n4 = KA4<<32|n4;

  if (Blen > LMEM_SIZE/4/2 || Alen > LMEM_SIZE/4/2 || Clen > LMEM_SIZE/4)
    printf("   GEMM00  m=%d n=%d ka=%d(/H) outloop[m/NCHIP/RMGRP*ka/H]=%d inloop[RMGRP*N/W]=%d Blen=%d/%d Alen=%d/%d Clen=%d/%d\n",
	   (Uint)m, (Uint)n, (Uint)ka, (Uint)(m/NCHIP/RMGRP*ka/H), (Uint)(RMGRP*N/W), (Uint)Blen, LMEM_SIZE/4/2, (Uint)Alen, LMEM_SIZE/4/2, (Uint)Clen, LMEM_SIZE/4);

  for (top=0; top<m/NCHIP; top+=RMGRP) { /* will be parallelized by multi-chip (M/#chip) */
    Force = 1;
    for (blk=0; blk<ka; blk+=H) { /* 3½Å¥ë¡¼¥×Å¸³«¤Î³°Â¦ÂÐ¾Ý */
      typedef struct {Uint i[4]} Ui4;
      Uint *a0[NCHIP];
      Uint *a[H][NCHIP];
      Ui4  *b[H], *b0[H], *b1[H], *b2[H], *b3[H];
      Ui4  *c0[NCHIP];
      Ui4  *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
      for (k=0; k<H; k++) {
	b[k] = i_m0B[LANE]+(blk+k)*n; b0[k] = b[k]; b1[k] = (Uint*)b[k]+1; b2[k] = (Uint*)b[k]+2;  b3[k] = (Uint*)b[k]+3; 
      }
      for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
	a0[CHIP] = i_m0A[LANE]+(CHIP*m/NCHIP+top)*ka;
	for (k=0; k<H; k++)
	  a[k][CHIP] = a0[CHIP]+blk+k;
	c0[CHIP] = i_m0C[LANE]+(CHIP*m/NCHIP+top)*n;
	c00[CHIP]= (Uint*)c0[CHIP]+0; c01[CHIP]= (Uint*)c0[CHIP]+1; c02[CHIP]= (Uint*)c0[CHIP]+2; c03[CHIP]= (Uint*)c0[CHIP]+3;
      }
      cofslimit1 = n4- 4; /* cofs32 < 36 x */
      cofslimit2 = n4- 8; /* cofs32 < 32 x */
      cofslimit3 = n4-12; /* cofs32 < 28 x */

#define sgemm00_40_core1(r, rm1, rp1) \
	    mop(OP_LDWR,   1, &BR[r][0][1],  (Ull)b0[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][0][0],  (Ull)b1[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][1][1],  (Ull)b2[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][1][0],  (Ull)b3[rm1], (Ull)cofs, MSK_W1, (Ull)b[rm1], Blen, 0, 0, (Ull)NULL, Blen);\
	    mop(OP_LDWR,   1, &BR[r][2][1],  (Ull)a[rm1][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);\
	    exe(OP_FMA, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FMA, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[r][2][1], EXP_H3210, BR[r][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

#define sgemm00_40_final(r, rp1, Force) \
	    exe(OP_CMP_LT,   &cc1, cofs, EXP_H3210, cofslimit1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_CMP_LT,   &cc2, cofs, EXP_H3210, cofslimit2, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_CMP_LT,   &cc3, cofs, EXP_H3210, cofslimit3, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    mop(OP_LDWR,   1, &BR[rp1][0][1],  (Ull)c00[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][1][1],  (Ull)c01[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][2][1],  (Ull)c02[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    mop(OP_LDWR,   1, &BR[rp1][3][1],  (Ull)c03[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    exe(OP_FAD, &AR[rp1][0], AR[r][0], EXP_H3210,  BR[rp1][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][1], AR[r][1], EXP_H3210,  BR[rp1][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][2], AR[r][2], EXP_H3210,  BR[rp1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    exe(OP_FAD, &AR[rp1][3], AR[r][3], EXP_H3210,  BR[rp1][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
	    mop(OP_STWR,   1, &AR[rp1][0],     (Ull)oofs, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex1,   0, 0, 0, cc1, 0xaaaa);\
	    mop(OP_STWR, ex1, &AR[rp1][1],     (Ull)oofs, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex2,   0, 0, 0, cc2, 0xaaaa);\
	    mop(OP_STWR, ex2, &AR[rp1][2],     (Ull)oofs, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen);\
	    cex(OP_CEXE,      &ex3,   0, 0, 0, cc3, 0xaaaa);\
	    mop(OP_STWR, ex3, &AR[rp1][3],     (Ull)oofs, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], Clen, 0, Force, (Ull)NULL, Clen)

//EMAX5A begin sgemm00_40 mapdist=0
/*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
  /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-KA4)<<32|((0-n4)&0xffffffff); LOOP1--; INIT1=0) { /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
    /*1*/ for (INIT0=1,LOOP0=N/W,cofs=(0-W*4)<<32|((0-W*4)&0xffffffff); LOOP0--; INIT0=0) {  /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
	    exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (W*4)<<32|(W*4), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);/* stage#0 */
	    exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?KA4n4:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* stage#0 */
	    exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffff, OP_NOP, 0LL);           /* stage#1 */

	    mop(OP_LDWR,   1, &BR[1][0][1],  (Ull)b0[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][0][0],  (Ull)b1[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][1][1],  (Ull)b2[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 */
	    mop(OP_LDWR,   1, &BR[1][1][0],  (Ull)b3[0], (Ull)cofs, MSK_W1, (Ull)b[0], Blen, 0, 0, (Ull)NULL, Blen);          /* stage#1 2KB */
	    mop(OP_LDWR,   1, &BR[1][2][1],  (Ull)a[0][CHIP],  (Ull)rofs, MSK_W1, (Ull)a0[CHIP], Alen, 0, 0, (Ull)NULL, Alen);/* stage#1 16KB */
	    exe(OP_FML, &AR[2][0], BR[1][0][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][1], BR[1][0][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][2], BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */
	    exe(OP_FML, &AR[2][3], BR[1][1][0], EXP_H3210,  BR[1][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);/* stage#2 */

	    sgemm00_40_core1( 2,  1,  3);
	    sgemm00_40_core1( 3,  2,  4);
	    sgemm00_40_core1( 4,  3,  5);
	    sgemm00_40_core1( 5,  4,  6);
	    sgemm00_40_core1( 6,  5,  7);
	    sgemm00_40_core1( 7,  6,  8);
	    sgemm00_40_core1( 8,  7,  9);
	    sgemm00_40_core1( 9,  8, 10);
	    sgemm00_40_core1(10,  9, 11);
	    sgemm00_40_core1(11, 10, 12);
	    sgemm00_40_core1(12, 11, 13);
	    sgemm00_40_core1(13, 12, 14);
	    sgemm00_40_core1(14, 13, 15);
	    sgemm00_40_core1(15, 14, 16);
	    sgemm00_40_core1(16, 15, 17);
	    sgemm00_40_core1(17, 16, 18);
	    sgemm00_40_core1(18, 17, 19);
	    sgemm00_40_core1(19, 18, 20);
	    sgemm00_40_core1(20, 19, 21);
#if (H==20)
	    sgemm00_40_final(21,     23, Force);
#endif
#if (H>20)
	    sgemm00_40_core1(21, 20, 22);
	    sgemm00_40_core1(22, 21, 23);
	    sgemm00_40_core1(23, 22, 24);
	    sgemm00_40_core1(24, 23, 25);
	    sgemm00_40_core1(25, 24, 26);
	    sgemm00_40_core1(26, 25, 27);
	    sgemm00_40_core1(27, 26, 28);
	    sgemm00_40_core1(28, 27, 29);
	    sgemm00_40_core1(29, 28, 30);
	    sgemm00_40_core1(30, 29, 31);
	    sgemm00_40_core1(31, 30, 32);
	    sgemm00_40_core1(32, 31, 33);
	    sgemm00_40_core1(33, 32, 34);
	    sgemm00_40_core1(34, 33, 35);
	    sgemm00_40_core1(35, 34, 36);
	    sgemm00_40_core1(36, 35, 37);
	    sgemm00_40_core1(37, 36, 38);
	    sgemm00_40_core1(38, 37, 39);
	    sgemm00_40_core1(39, 38, 40);
	    sgemm00_40_core1(40, 39, 41);
#endif
#if (H==40)
	    /****final*****/
	    sgemm00_40_final(41,     43, Force);
#endif
          }
        }
      }
//EMAX5A end
      if (Force) Force = 0; /* reset wdat load to LMM */
printf("*");
    }
  }
//EMAX5A drain_dirty_lmm
  monitor_time_start(THREAD, IMAX_CPYOUT);
  xmax_cpyout(2, C, 1, 1, i_m0C[LANE], m, n, n); /* i_m0C is contiguous w/ CEX+ST */
  monitor_time_end(THREAD, IMAX_CPYOUT);
}

void xmax_sgemm10(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  int row, col, k;

#if defined(CBLAS_GEMM)
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, ka, 1.0f, A, m, B, n, 0.0f, C, n);
#else
  for (k=0; k<ka; k++) {
    for (row=0; row<m; row++) {
      for (col=0; col<n; col++) {
	if (k==0) C[row*n+col]  = A[k*m+row] * B[k*n+col];
	else      C[row*n+col] += A[k*m+row] * B[k*n+col];
      }
    }
  }
#endif

  /* ¡ú¡ú¡ú PBL1-2 ¡ú¡ú¡ú */
}

void xmax_sgemm01(int THREAD, int LANE, int m, int n, int ka, float *A, float *B, float *C) /* C=A*B */
{
  int row, col, k;

#if defined(CBLAS_GEMM)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, ka, 1.0f, A, ka, B, ka, 0.0f, C, n);
#else
  for (row=0; row<m; row++) {
    for (col=0; col<n; col++) {
      for (k=0; k<ka; k++) {
	if (k==0) C[row*n+col]  = A[row*ka+k] * B[col*ka+k];
	else      C[row*n+col] += A[row*ka+k] * B[col*ka+k];
      }
    }
  }
#endif

  /* ¡ú¡ú¡ú PBL1-3 ¡ú¡ú¡ú */
}

void xmax_conv_backward(int THREAD, int LANE, float4D *out, float2D *kernel, float2D *g_kernel, float4D *in, int ksize)
{
  int   kstride = 1;
  int   BATCH  = in->nstrides;  //100
  int   IC     = in->nchannel;  //3
  int   IM     = in->kstrides;  //28
  int   IMX;
  int   OC     = out->nchannel; //8
  int   M      = out->kstrides; //24
  int   K      = ksize;         // 5,4,3,2,1
  Uint  *in0   = in->data;      // IC*IM*IM
  Uint  *ker   = kernel->data;  // OC*IC*K*K
  Uint  *g_ker = g_kernel->data;// OC*IC*K*K
  Uint  *out0  = out->data;     // OC*M*M
  Uint  *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *op0, *kp, kidx, *kp0;
  int   pad;
  int   count, top, iset, oset, oc, w, ic, y, x;
  int   y0, x0, ch, xy;
  Ull   IMX4, IM4, M4, IMX4M4, M4IM4, IMXlen, IMlen, Mlen;
  Ull   CHIP, img, rofs, cofs, iofs, oofs, b00, c00;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   cc0, cc1, cc2, cc3, ex0, ex1;
  Ull   Force;

  /*  unpack_patch2col(tmp_col, in, ksize, kstride, M, M); */
  /*  for (oc=0;oc<OC;oc++) {                              */
  /*    for (img=0;img<BATCH;img++)                        */
  /*      memcpy(&(tmp_dst->data[(oc*BATCH+img)*M*M]), &(out->data[(oc+img*OC)*M*M]), M*M*sizeof(float));*/
  /*  }                                                    */
  /*  multiply_float2D(g_kernel, tmp_dst, 0, tmp_col, 1); // 8x25 dot 25x57600 --> 8x57600 */
  /*  multiply_float2D(tmp_col, kernel, 1, tmp_dst, 0);    */
  /*  pack_col2patch(in, tmp_col, ksize, kstride, M, M);   */

  if (IM == M)
    pad = 0;   /* PADÌµ¤·.in¼þ°Ï0.0¤ò²¾Äê */
  else if ((IM - K)/1 + 1 == M)
    pad = K/2; /* PADÍ­¤ê.in¼þ°ÏÆÃÊÌ°·¤¤ÉÔÍ× */
  else {
    printf("xmax_conv_backward error: IM=%d K=%d M=%d\n", IM, K, M);
    printf("IM == M || (IM-K)/1+1 == M\n");
    exit(-1);
  }

  /*================================================================================================*/
  /*=== back_g_ker =================================================================================*/
  /*================================================================================================*/

#undef  IMAP
#undef  OMAP
#undef  W
#undef  NCHIP
#define IMAP  2
#define OMAP  16
#define W     1
#define NCHIP 1

  /***********************************/
  /* ¡ú¡ú¡ú PBL1-4 (g_kernel) ¡ú¡ú¡ú */
  /***********************************/
#if 0
  xmax_cpyin(2, i_outX[LANE], &M,  out0, BATCH, OC,  M, M, 1); //dst[OC][M][BATCH][M]     <- src[BATCH][OC][M][M]
  xmax_cpyin(2, i_inpX[LANE], &IMX, in0, BATCH, IC, IM, M, K); //dst[IC][IMX][BATCH][IMX] <- src[BATCH][IC][IM][IM]
  xmax_bzero(i_kerX[LANE], OC*IC*K*K); /* g_kernel */
  for (oset=0; oset<((OC+OMAP-1)&~(OMAP-1)); oset+=OMAP) { /* set output channel */
    Uint *ip0[IMAP], *op0[OMAP], *kp0[OMAP][IMAP];
    for (rofs=0; rofs<M; rofs++) {
      for (iset=0; iset<((IC+IMAP-1)&~(IMAP-1)); iset+=IMAP) { /* set offset of input channel */
	kidx = 0;
	for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
	  for (x=-(K/2); x<K-(K/2); x++) {
	    for (ic=0; ic<IMAP; ic++)
	      ip0[ic] = (iset+ic)<IC ? &i_inpX[LANE][(iset+ic)*IMX*BATCH*IMX+(rofs+y+K/2)*BATCH*IMX+(x+K/2)] : 0; /* input */
	    for (oc=0; oc<OMAP; oc++)
	      op0[oc] = (oset+oc)<OC ? &i_outX[LANE][(oset+oc)*M*BATCH*M+rofs*BATCH*M] : 0; /* output */
	    for (ic=0; ic<IMAP; ic++) {
	      for (oc=0; oc<OMAP; oc++)
		kp0[oc][ic] = ((iset+ic)<IC && (oset+oc)<OC) ? &i_kerX[LANE][((oset+oc)*IC+iset+ic)*K*K+kidx] : 0; /* NULL skip DMA */
	    }
	    for (ic=0; ic<IMAP&&(iset+ic)<IC; ic++) { /* set output channel */
	      for (oc=0; oc<OMAP&&(oset+oc)<OC; oc++) { /* set output channel */
		for (img=0; img<BATCH; img++) {
		  for (cofs=0; cofs<M; cofs++) { /* image loop (cofs) */
		    float in  = *(float*)&ip0[ic][img*IMX+cofs];
		    float out = *(float*)&op0[oc][img*M+cofs];
		    *(float*)kp0[oc][ic] += out * in;
		  }
		}
	      }
	    }
	    kidx++;
	  }
        }
      }
    }
  }
  xmax_cpyout(2, g_ker, 1, 1, i_kerX[LANE], IC*K*K, OC, OC); /* g_kernel */
#else
  monitor_time_start(THREAD, IMAX_CPYIN);
  xmax_cpyin(2, i_out[LANE], &M,  out0, BATCH, OC,  M, M, 1); //dst[OC][M][BATCH][M]     <- src[BATCH][OC][M][M]
  xmax_cpyin(2, i_inp[LANE], &IMX, in0, BATCH, IC, IM, M, K); //dst[IC][IMX][BATCH][IMX] <- src[BATCH][IC][IM][IM]
  xmax_bzero(i_ker[LANE], OC*IC*K*K); /* g_kernel */
  monitor_time_end(THREAD, IMAX_CPYIN);
  IMX4   = IMX*4;
  M4     = M*4;
  IMX4M4 = IMX4<<32|M4;
  IMXlen = IMX*BATCH;
  Mlen   = M*BATCH;

  if (IMXlen > LMEM_SIZE/4/2 || Mlen > LMEM_SIZE/4/4)
    printf("   BACK00  IMXlen=%d/%d Mlen=%d/%d\n", (Uint)IMXlen, LMEM_SIZE/4/2, (Uint)Mlen, LMEM_SIZE/4/4);

  /* +----------------------+-----------------------+                     */
  /* |     inp[ic][row]     |out[oc+0][row+yx*]ºÆÍøÍÑ K¹Ô                 */
  /* |                      |ker[oc+0][ic][yx*]     |                     */
  /* +----------------------+-----------------------+                     */
  /* |     inp[ic][row]     |out[oc+1][row+yx*]ºÆÍøÍÑ K¹Ô                 */
  /* |                      |ker[oc+1][ic][yx*]     |                     */
  /* +----------------------+-----------------------+                     */
  /* |     inp[ic][row]     |out[oc+2][row+yx*]ºÆÍøÍÑ K¹Ô                 */
  /* |                      |ker[oc+2][ic][yx*]     |                     */
  /* +----------------------+-----------------------+                     */
  /* |     inp[ic][row]     |out[oc+3][row+yx*]ºÆÍøÍÑ K¹Ô                 */
  /* |                      |ker[oc+3][ic][yx*]     |                     */
  /* +----------------------+-----------------------+                     */
  /*                             oc:stage¤ËÅ¸³«                           */
  /*                                   ic:ºÇ³°¥ë¡¼¥×                      */
  /*                                       y:ÃÊ¿ô¤òËä¤á¤ë¤Û¤ÉÂ¿¤¯¤Ê¤¤     */
  /*                                        x:ÎÙÀÜÍ×ÁÇ¤ÏÊ£¿ôLMM¤ËÊ¬»¶ÉÔ²Ä */
  for (oset=0; oset<((OC+OMAP-1)&~(OMAP-1)); oset+=OMAP) { /* set output channel */
    Ull  cc0[OMAP][IMAP], cc1[OMAP][IMAP];
    Uint inum[IMAP][NCHIP], *ip0[IMAP][NCHIP], *it0[IMAP][NCHIP];
    Uint onum[OMAP], *op0[OMAP], *ot0[OMAP];
    Uint *kp0[OMAP][IMAP][NCHIP];
    for (iset=0; iset<((IC+IMAP*NCHIP-1)&~(IMAP*NCHIP-1)); iset+=IMAP*NCHIP) { /* set offset of input channel */
      for (rofs=0; rofs<M; rofs++) {
	kidx = 0;
	for (y=-(K/2); y<K-(K/2); y++) { /* kernel loop */
	  for (x=-(K/2); x<K-(K/2); x++) {
	    for (CHIP=0; CHIP<NCHIP; CHIP++) {
	      for (ic=0; ic<IMAP; ic++) {
		inum[ic][CHIP] = iset+IMAP*CHIP+ic;
		ip0[ic][CHIP]  = (iset+IMAP*CHIP+ic)<IC ? &i_inp[LANE][(iset+IMAP*CHIP+ic)*IMX*BATCH*IMX+(rofs+y+K/2)*BATCH*IMX+(x+K/2)] : 0; /* input */
		if (IMX*BATCH*IMX <= 32768/4) {
		  IMXlen = IMX*BATCH*IMX;
		  it0[ic][CHIP] = (iset+IMAP*CHIP+ic)<IC ? &i_inp[LANE][(iset+IMAP*CHIP+ic)*IMX*BATCH*IMX                       ] : 0;         /* input */
		}
		else if (IMX*BATCH*K <= 32768/4) {
		  IMXlen = IMX*BATCH*K;
		  it0[ic][CHIP] = (iset+IMAP*CHIP+ic)<IC ? &i_inp[LANE][(iset+IMAP*CHIP+ic)*IMX*BATCH*IMX+(rofs      )*BATCH*IMX] : 0;         /* input */
		}
		else
		  it0[ic][CHIP] = (iset+IMAP*CHIP+ic)<IC ? &i_inp[LANE][(iset+IMAP*CHIP+ic)*IMX*BATCH*IMX+(rofs+y+K/2)*BATCH*IMX] : 0;         /* input */
	      }
	    }
	    for (oc=0; oc<OMAP; oc++) {
	      onum[oc] = oset+oc;
	      op0[oc]  = (oset+oc)<OC ? &i_out[LANE][(oset+oc)*M*BATCH*M+rofs*BATCH*M] : 0; /* output */
	      if (M*BATCH*M <= 16384/4) {
		Mlen = M*BATCH*M;
		ot0[oc] = (oset+oc)<OC ? &i_out[LANE][(oset+oc)*M*BATCH*M] : 0; /* output */
	      }
	      else
		ot0[oc] = op0[oc];
	    }
	    for (oc=0; oc<OMAP; oc++) {
	      for (CHIP=0; CHIP<NCHIP; CHIP++) {
		for (ic=0; ic<IMAP; ic++)
		  kp0[oc][ic][CHIP] = ((iset+IMAP*CHIP+ic)<IC && (oset+oc)<OC) ? &i_ker[LANE][((oset+oc)*IC+iset+IMAP*CHIP+ic)*K*K+kidx] : 0; /* NULL skip DMA */
	      }
	    }
	    Force = 1;

#define back_g_ker_core1(b, o, i, Force) \
  exe(OP_CMP_LT,   &cc0[o][i],onum[o],       EXP_H3210,            OC,          EXP_H3210, 0LL,                  EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#1 */\
  exe(OP_CMP_LT,   &cc1[o][i],inum[i][CHIP], EXP_H3210,            IC,          EXP_H3210, 0LL,                  EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#1 */\
  mop(OP_LDWR,  1, &BR[b][1][1],             (Ull)op0[o],          oofs,        MSK_W0,    (Ull)ot0[o],          Mlen,      0,      0,   NULL,   Mlen);   /* stage#2 */\
  mop(OP_LDWR,  1, &BR[b][2][1],             (Ull)ip0[i][CHIP],    iofs,        MSK_W1,    (Ull)it0[i][CHIP],    IMXlen,    0,      0,   NULL,   IMXlen); /* stage#2 IMXlen¤¬Âç¤­¤¤¤Î¤ÇLMM*2»ÈÍÑ */\
  exe(OP_NOP,      &AR[b][0], 0LL,           EXP_H3210,            0LL,         EXP_H3210, 0LL,                  EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#2 (dummy to set target location) */\
  mop(OP_LDWR,  1, &b00,                     (Ull)kp0[o][i][CHIP], 0LL,         MSK_W0,    (Ull)kp0[o][i][CHIP], 1LL,       0,      Force, NULL, 1LL);    /* stage#2 fold¤Ïunit[0]¤ËÍ×»ØÄê */\
  exe(OP_FMA,      &b00,      b00,           EXP_H3210,            BR[b][2][1], EXP_H3210, BR[b][1][1],          EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);    /* stage#2 */\
  cex(OP_CEXE,     &ex0, 0, 0, cc1[o][i], cc0[o][i], 0x8888);                                                                                             /* stage#2 */\
  mop(OP_STWR,ex0, &b00,                     (Ull)kp0[o][i][CHIP], 0LL,         MSK_D0,    (Ull)kp0[o][i][CHIP], 1LL,       0,      Force, NULL, 1LL)     /* stage#2 */

//EMAX5A begin back_g_ker mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
        /*2*/ for (INIT1=1,LOOP1=BATCH,img=(0-IMX4)<<32|((0-M4)&0xffffffff); LOOP1--; INIT1=0) {                           /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                            /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,      &img,  img,             EXP_H3210,  INIT0?IMX4M4:0, EXP_H3210,  0LL, EXP_H3210, OP_NOP,   0LL,                  OP_NOP, 0LL); /* stage#0 */
		  exe(OP_ADD,      &cofs, INIT0?cofs:cofs, EXP_H3210,  4LL<<32|4LL,    EXP_H3210,  0LL, EXP_H3210, OP_AND,   0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
		  exe(OP_ADD,      &iofs, img,             EXP_H3210,  cofs,           EXP_H3210,  0LL, EXP_H3210, OP_AND,   0xffffffff00000000LL, OP_NOP, 0LL); /* stage#1 */
		  exe(OP_ADD,      &oofs, img,             EXP_H3210,  cofs,           EXP_H3210,  0LL, EXP_H3210, OP_AND,   0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#1 */

		  back_g_ker_core1( 2,  0,  0, Force); /**** oc0  ic0*****/
		  back_g_ker_core1( 3,  0,  1, Force); /**** oc0  ic1*****/
		  back_g_ker_core1( 4,  1,  0, Force); /**** oc1  ic0*****/
		  back_g_ker_core1( 5,  1,  1, Force); /**** oc1  ic1*****/
		  back_g_ker_core1( 6,  2,  0, Force); /**** oc2  ic0*****/
		  back_g_ker_core1( 7,  2,  1, Force); /**** oc2  ic1*****/
		  back_g_ker_core1( 8,  3,  0, Force); /**** oc3  ic0*****/
		  back_g_ker_core1( 9,  3,  1, Force); /**** oc3  ic1*****/
		  back_g_ker_core1(10,  4,  0, Force); /**** oc4  ic0*****/
		  back_g_ker_core1(11,  4,  1, Force); /**** oc4  ic1*****/
		  back_g_ker_core1(12,  5,  0, Force); /**** oc5  ic0*****/
		  back_g_ker_core1(13,  5,  1, Force); /**** oc5  ic1*****/
		  back_g_ker_core1(14,  6,  0, Force); /**** oc6  ic0*****/
		  back_g_ker_core1(15,  6,  1, Force); /**** oc6  ic1*****/
		  back_g_ker_core1(16,  7,  0, Force); /**** oc7  ic0*****/
		  back_g_ker_core1(17,  7,  1, Force); /**** oc7  ic1*****/
		  back_g_ker_core1(18,  8,  0, Force); /**** oc8  ic0*****/
		  back_g_ker_core1(19,  8,  1, Force); /**** oc8  ic1*****/
		  back_g_ker_core1(20,  9,  0, Force); /**** oc9  ic0*****/
		  back_g_ker_core1(21,  9,  1, Force); /**** oc9  ic1*****/
		  back_g_ker_core1(22, 10,  0, Force); /**** oc10 ic0*****/
		  back_g_ker_core1(23, 10,  1, Force); /**** oc10 ic1*****/
		  back_g_ker_core1(24, 11,  0, Force); /**** oc11 ic0*****/
		  back_g_ker_core1(25, 11,  1, Force); /**** oc11 ic1*****/
		  back_g_ker_core1(26, 12,  0, Force); /**** oc12 ic0*****/
		  back_g_ker_core1(27, 12,  1, Force); /**** oc12 ic1*****/
		  back_g_ker_core1(28, 13,  0, Force); /**** oc13 ic0*****/
		  back_g_ker_core1(29, 13,  1, Force); /**** oc13 ic1*****/
		  back_g_ker_core1(30, 14,  0, Force); /**** oc14 ic0*****/
		  back_g_ker_core1(31, 14,  1, Force); /**** oc14 ic1*****/
		  back_g_ker_core1(32, 15,  0, Force); /**** oc15 ic0*****/
		  back_g_ker_core1(33, 15,  1, Force); /**** oc15 ic1*****/
                }
              }
            }
//EMAX5A end
            if (Force) Force = 0; /* reset wdat load to LMM */
            kidx++;
          }
        }
      }
    }
  }
//EMAX5A drain_dirty_lmm
  monitor_time_start(THREAD, IMAX_CPYOUT);
  xmax_cpyout(2, g_ker, 1, 1, i_ker[LANE], IC*K*K, OC, OC); /* g_kernel */
  monitor_time_end(THREAD, IMAX_CPYOUT);
#endif

  /*================================================================================================*/
  /*=== back_in ====================================================================================*/
  /*================================================================================================*/

#undef  IMAP
#undef  OMAP
#undef  W
#undef  NCHIP
#define IMAP  2
#define OMAP  16
#define W     1
#define NCHIP 1

  /***********************************/
  /* ¡ú¡ú¡ú PBL1-5 (in)       ¡ú¡ú¡ú */
  /***********************************/
#if 0
  xmax_cpyin(2, i_outX[LANE], &M, out0, BATCH, OC, M, M, 1); //dst[OC][M][BATCH][M] <- src[BATCH][OC][M][M]
  xmax_cpyin(0, i_kerX[LANE], &K, ker,  OC,    IC, K, K, 1); //imemcpy(i_ker[LANE], ker,  OC*IC*K*K); K=K;
  xmax_bzero(i_inpX[LANE], IC*IM*BATCH*IM); /* in */
  if (K == 1 || IM-K+1 == M) { y0 = 0;    x0 = 0;    }
  else if (IM == M)          { y0 = -K/2; x0 = -K/2; }
  for (oset=0; oset<((OC+OMAP-1)&~(OMAP-1)); oset+=OMAP) { /* set output channel */
    Uint *op0[OMAP]; float kp0[OMAP];
    for (rofs=0;rofs<M;rofs++) { /*24, 10*/
      for (ch=0;ch<IC*K*K;ch++) { /*5x5, 8x3x3*/
	ic = ch/(K*K);
	y  = ch%(K*K)/K + y0;
	x  = ch%(K*K)%K + x0;
	if (0<=rofs+y && rofs+y<IM) {
	  ip0 = &i_inpX[LANE][ic*IM*BATCH*IM+(rofs+y)*BATCH*IM+x];
	  for (oc=0; oc<OMAP&&(oset+oc)<OC; oc++) { /* set output channel */
	    op0[oc] = &i_outX[LANE][(oset+oc)*M*BATCH*M+rofs*BATCH*M];
	    kp0[oc] = *(float*)&i_kerX[LANE][(oset+oc)*IC*K*K+ch];
	    for (img=0;img<BATCH;img++) { /*100, 100*/
	      for (cofs=0;cofs<M;cofs++) { /*24, 10*/
		if (0<=cofs+x && cofs+x<IM) {
		  *(float*)&ip0[img*IM+cofs] += kp0[oc] * *(float*)&op0[oc][img*M+cofs];
		}
	      }
	    }
	  }
	}
      }
    }
  }
  xmax_cpyout(1, in0, BATCH, IC, i_inpX[LANE], IM, IM, IC); /* in */
#else
  monitor_time_start(THREAD, IMAX_CPYIN);
//xmax_cpyin(2, i_out[LANE], &M, out0, BATCH, OC, M, M, 1); //dst[OC][M][BATCH][M] <- src[BATCH][OC][M][M]
  xmax_cpyin(0, i_ker[LANE], &K, ker,  OC,    IC, K, K, 1); //imemcpy(i_ker[LANE], ker,  OC*IC*K*K); K=K;
  xmax_bzero(i_inp[LANE], IC*IM*BATCH*IM); /* in */
  monitor_time_end(THREAD, IMAX_CPYIN);
  IM4    = IM*4;
  M4     = M*4;
  M4IM4  = M4<<32|IM4;
  IMlen  = IM*BATCH;
  Mlen   = M*BATCH;
  if (K == 1 || IM-K+1 == M) { y0 = 0;    x0 = 0;    }
  else if (IM == M)          { y0 = -K/2; x0 = -K/2; }

  if (IMlen > LMEM_SIZE/4 || Mlen > LMEM_SIZE/4)
    printf("   BACK10  IMlen=%d/%d Mlen=%d/%d\n", (Uint)IMlen, LMEM_SIZE/4, (Uint)Mlen, LMEM_SIZE/4);

  /* +----------------------+-----------------------+                     */
  /* |   ker[oc+0][ic][yx]  |out[oc+0][row+yx*]ºÆÍøÍÑ K¹Ô                 */
  /* +----------------------+-----------------------+                     */
  /* |   ker[oc+1][ic][yx]  |out[oc+1][row+yx*]ºÆÍøÍÑ K¹Ô                 */
  /* +----------------------+-----------------------+                     */
  /* |   ker[oc+2][ic][yx]  |out[oc+2][row+yx*]ºÆÍøÍÑ K¹Ô                 */
  /* +----------------------+-----------------------+                     */
  /* |   ker[oc+3][ic][yx]  |out[oc+3][row+yx*]ºÆÍøÍÑ K¹Ô                 */
  /* |                      |inp[ic][row]           |                     */
  /* +----------------------+-----------------------+                     */
  /*                             oc:stage¤ËÅ¸³«                           */
  /*                             ic:ºÇ³°¥ë¡¼¥×                            */
  /*                                       y:ÃÊ¿ô¤òËä¤á¤ë¤Û¤ÉÂ¿¤¯¤Ê¤¤     */
  /*                                        x:¹ÔÊý¸þ                      */
  for (oset=0; oset<((OC+OMAP-1)&~(OMAP-1)); oset+=OMAP) { /* set output channel */
    Uint inum[IMAP][NCHIP], *ip0[IMAP][NCHIP], *it0[IMAP][NCHIP];
    Uint onum[OMAP], *op0[OMAP], *ot0[OMAP];
    Uint kp0[OMAP][IMAP][NCHIP];
    for (iset=0; iset<((IC+IMAP*NCHIP-1)&~(IMAP*NCHIP-1)); iset+=IMAP*NCHIP) { /* set offset of input channel */
      for (rofs=0;rofs<M;rofs++) { /*24, 10*/
        for (xy=0;xy<K*K;xy++) { /*5x5, 8x3x3*/
          y  = xy/K + y0;
          x  = xy%K + x0;
          Ull  yIM4  = y*IM4;
          Ull  x4    = x*4;
          Ull  IMIM4 = IM*IM4;
          if (xy%K == 0) Force = 1;
	  if (0<=rofs+y && rofs+y<IM) {
	    for (CHIP=0; CHIP<NCHIP; CHIP++) {
	      for (ic=0; ic<IMAP; ic++) {
		inum[ic][CHIP] = iset+IMAP*CHIP+ic;
		ip0[ic][CHIP]  = (iset+IMAP*CHIP+ic)<IC ? &i_inp[LANE][(iset+IMAP*CHIP+ic)*IM*BATCH*IM+(rofs+y)*BATCH*IM+x] : 0;
		it0[ic][CHIP]  = (iset+IMAP*CHIP+ic)<IC ? &i_inp[LANE][(iset+IMAP*CHIP+ic)*IM*BATCH*IM+(rofs+y)*BATCH*IM] : 0; // x¤Î¥Þ¥¤¥Ê¥¹À®Ê¬¤ò½üµî
	      }
	    }
	    for (oc=0; oc<OMAP; oc++) {
	      onum[oc] = oset+oc;
	      op0[oc]  = (oset+oc)<OC ? &i_out[LANE][(oset+oc)*M*BATCH*M  +rofs*BATCH*M] : 0;
	      if (M*BATCH*M <= LMEM_SIZE/4) {
		Mlen = M*BATCH*M;
		ot0[oc] = (oset+oc)<OC ? &i_out[LANE][(oset+oc)*M*BATCH*M] : 0; /* output */
	      }
	      else
		ot0[oc] = op0[oc];
	    }
	    for (oc=0; oc<OMAP; oc++) {
	      for (CHIP=0; CHIP<NCHIP; CHIP++) {
		for (ic=0; ic<IMAP; ic++)
		  kp0[oc][ic][CHIP]  = ((oset+oc)<OC && (iset+IMAP*CHIP+ic)<IC) ? i_ker[LANE][(oset+oc)*IC*K*K+(iset+IMAP*CHIP+ic)*K*K+xy] : 0; /* 0.0 */
	      }
	    }

#define back_in_core1(b, bp1, o, i) \
  mop(OP_LDWR,  1, &BR[b][0][1],          (Ull)op0[o], oofs,            MSK_W1,    (Ull)ot0[o], Mlen,      0,      0,   NULL,   Mlen); /* stage#2 */\
  exe(OP_FMA,      &AR[bp1][0], AR[b][0], EXP_H3210,   kp0[o][i][CHIP], EXP_H3210, BR[b][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)   /* stage#3 */

#define back_in_final(b, bp2, i, Force) \
  exe(OP_ADD,      &r10,      cofs,            EXP_H3210,         x4,                  EXP_H3210, 0LL,               EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);   /* stage#5 */\
  exe(OP_CMP_LT,   &cc0,      r10,             EXP_H3210,         IM4,                 EXP_H3210, 0LL,               EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#6 */\
  mop(OP_LDWR,  1, &BR[bp2][0][1],             (Ull)ip0[i][CHIP], iofs,                MSK_W0,    (Ull)it0[i][CHIP], IMlen,     0,      Force,                NULL,   IMlen); /* stage#7 */\
  exe(OP_FAD,      &AR[bp2][0], AR[b][0],      EXP_H3210,         BR[bp2][0][1],       EXP_H3210, 0LL,               EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#7 */\
  cex(OP_CEXE,     &ex0, 0, 0, 0, cc0, 0xaaaa);                                                                                                                               /* stage#7 */\
  mop(OP_STWR,ex0, &AR[bp2][0],                iofs,              (Ull)ip0[i][CHIP],   MSK_D0,    (Ull)it0[i][CHIP], IMlen,     0,      Force,                NULL,   IMlen)  /* stage#7 */

//EMAX5A begin back_in mapdist=0
      /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* output channels are parallelized by multi-chip (OC4/#chip) */
	/*2*/ for (INIT1=1,LOOP1=BATCH,img=(0-M4)<<32|((0-IM4)&0xffffffff); LOOP1--; INIT1=0) {                                       /* mapped to FOR() on BR[63][1][0] */ /* stage#0 */
          /*1*/ for (INIT0=1,LOOP0=M,cofs=(0-4LL)<<32|((0-4LL)&0xffffffff); LOOP0--; INIT0=0) {                                       /* mapped to FOR() on BR[63][0][0] */ /* stage#0 */
                  exe(OP_ADD,      &img,      img,             EXP_H3210,   INIT0?M4IM4:0, EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#0 */
		  exe(OP_ADD,      &cofs,     INIT0?cofs:cofs, EXP_H3210,   4LL<<32|4LL,   EXP_H3210, 0LL,         EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);   /* stage#0 */
		  exe(OP_ADD,      &iofs,     img,             EXP_H3210,   cofs,          EXP_H3210, 0LL,         EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);   /* stage#1 */
		  exe(OP_ADD,      &oofs,     img,             EXP_H3210,   cofs,          EXP_H3210, 0LL,         EXP_H3210, OP_AND, 0xffffffff00000000LL, OP_NOP, 0LL);   /* stage#1 */

		  /****ic0*****/
		  mop(OP_LDWR,  1, &BR[2][0][1],                   (Ull)op0[0], oofs,          MSK_W1,    (Ull)ot0[0], Mlen,      0,      0,                    NULL,   Mlen);  /* stage#2 */
		  exe(OP_FML,      &AR[3][0],    kp0[0][0][CHIP],  EXP_H3210,   BR[2][0][1],   EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#3 */
		  back_in_core1( 3,  4,  1,  0); /**** oc1  ic0*****/
		  back_in_core1( 4,  5,  2,  0); /**** oc2  ic0*****/
		  back_in_core1( 5,  6,  3,  0); /**** oc3  ic0*****/
		  back_in_core1( 6,  7,  4,  0); /**** oc4  ic0*****/
		  back_in_core1( 7,  8,  5,  0); /**** oc5  ic0*****/
		  back_in_core1( 8,  9,  6,  0); /**** oc6  ic0*****/
		  back_in_core1( 9, 10,  7,  0); /**** oc7  ic0*****/
		  back_in_core1(10, 11,  8,  0); /**** oc8  ic0*****/
		  back_in_core1(11, 12,  9,  0); /**** oc9  ic0*****/
		  back_in_core1(12, 13, 10,  0); /**** oc10 ic0*****/
		  back_in_core1(13, 14, 11,  0); /**** oc11 ic0*****/
		  back_in_core1(14, 15, 12,  0); /**** oc12 ic0*****/
		  back_in_core1(15, 16, 13,  0); /**** oc13 ic0*****/
		  back_in_core1(16, 17, 14,  0); /**** oc14 ic0*****/
		  back_in_core1(17, 18, 15,  0); /**** oc15 ic0*****/
		  back_in_final(18, 20,  0, Force); /****OMAP(16)+2,OMAP(16)+4****/

		  /****ic1*****/
		  mop(OP_LDWR,  1, &BR[21][0][1],                  (Ull)op0[0], oofs,          MSK_W1,    (Ull)ot0[0], Mlen,      0,      0,                    NULL,   Mlen);  /* stage#2 */
		  exe(OP_FML,      &AR[22][0],   kp0[0][1][CHIP],  EXP_H3210,  BR[21][0][1],   EXP_H3210, 0LL,         EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);   /* stage#3 */
		  back_in_core1(22, 23,  1,  1); /**** oc1  ic1*****/
		  back_in_core1(23, 24,  2,  1); /**** oc2  ic1*****/
		  back_in_core1(24, 25,  3,  1); /**** oc3  ic1*****/
		  back_in_core1(25, 26,  4,  1); /**** oc4  ic1*****/
		  back_in_core1(26, 27,  5,  1); /**** oc5  ic1*****/
		  back_in_core1(27, 28,  6,  1); /**** oc6  ic1*****/
		  back_in_core1(28, 29,  7,  1); /**** oc7  ic1*****/
		  back_in_core1(29, 30,  8,  1); /**** oc8  ic1*****/
		  back_in_core1(30, 31,  9,  1); /**** oc9  ic1*****/
		  back_in_core1(31, 32, 10,  1); /**** oc10 ic1*****/
		  back_in_core1(32, 33, 11,  1); /**** oc11 ic1*****/
		  back_in_core1(33, 34, 12,  1); /**** oc12 ic1*****/
		  back_in_core1(34, 35, 13,  1); /**** oc13 ic1*****/
		  back_in_core1(35, 36, 14,  1); /**** oc14 ic1*****/
		  back_in_core1(36, 37, 15,  1); /**** oc15 ic1*****/
		  back_in_final(37, 39,  1, Force); /****OMAP(35)+2,OMAP(35)+4****/
	        }
	      }
	    }
//EMAX5A end
            if (Force) Force = 0; /* reset wdat load to LMM */
	  }
	}
      }
    }
  }
//EMAX5A drain_dirty_lmm
  monitor_time_start(THREAD, IMAX_CPYOUT);
  xmax_cpyout(1, in0, BATCH, IC, i_inp[LANE], IM, IM, IC); /* in */
  monitor_time_end(THREAD, IMAX_CPYOUT);
#endif
}

