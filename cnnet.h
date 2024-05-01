// Copyright 2014 <yaojun@is.naist.jp>

#ifndef CNNET_H_
#define CNNET_H_

#include "tensor.h"

struct c {
  int isize;  /* isize x isize */
  int ichan;  /* in_channels   */
  int ksize;  /* ksize x ksize */
  int osize;  /* osize x osize */
  int ochan;  /* out_channels  */
  int psize;  /* pooling_size  */
};

struct f {
  int osize;  /* osize x osize */
};

#define CNN_DEPTH_MAX 7
#define FC_DEPTH_MAX  7

typedef struct _CNNet {
  /* int oheight; */
  /* int owidth; */
  /* int nbatch; */
  /* int nchannel; */
  /* int ksize, kstrides, psize; */
  float4D ninput;
  float4D work;
  float4D attention;

  float4D ninput_dropoutmask[CNN_DEPTH_MAX]; /* DROPOUT */
  float2D tmp_col[CNN_DEPTH_MAX];
  float2D tmp_dst[CNN_DEPTH_MAX];
  float2D Ki2h[CNN_DEPTH_MAX];
  float2D g_Ki2h[CNN_DEPTH_MAX];
  float4D nhidden[CNN_DEPTH_MAX];
  float4D nhiddenbak[CNN_DEPTH_MAX];
  float2D hbias[CNN_DEPTH_MAX];
  float2D g_hbias[CNN_DEPTH_MAX];
  float4D npool[CNN_DEPTH_MAX];
  float4D npoolbak[CNN_DEPTH_MAX];

  float2D nflat_dropoutmask[FC_DEPTH_MAX]; /* DROPOUT */
  float2D nflat[FC_DEPTH_MAX];
  float2D Wh2o[FC_DEPTH_MAX];
  float2D g_Wh2o[FC_DEPTH_MAX];
  float2D nout[FC_DEPTH_MAX];
  float2D noutbak[FC_DEPTH_MAX];
  float2D obias[FC_DEPTH_MAX];
  float2D g_obias[FC_DEPTH_MAX];
} CNNet;

void init_net(CNNet*, int, struct c*, struct f*);
void reset_dropoutmask();
void reset_dropoutmask4D();
void reset_dropoutmask2D();
void show_net();
void unpack_patch2col();
void pack_col2patch();
void relu2();
void relu4();
void relu_grad_merge();
void max_pooling();
void max_unpooling();
void nn_forward(int, int, CNNet*, struct c*, struct f*, float2D*, int);
void conv_forward(int, int, float4D*, float2D*, float4D*, int, float2D*, float2D*);
void nn_backprop(int, CNNet*, struct c*, struct f*, float2D*, int);
void conv_backward(int, const float4D*, const float2D*, float2D*, float4D*, int, float2D*, float2D*);
void nn_update(int, CNNet*, float, float, float, int);
void smax_trial(int, CNNet*, struct c*, struct f*);
void smax_train(CNNet*, struct c*, struct f*);
void smax_update(CNNet*, float, float);

#define MAX_NTHREAD 16

volatile struct th_inference_args {
  int      thid;
  int      dmy0[15]; /* to isolate cache line */
  int      stat;     /* 0:idle, 1:run, 2:wait (enq/deq, DMA, EXEC) */
  int      dmy1[15]; /* to isolate cache line */
  sigset_t sigset;   /* sys/_sigset.h 2B/4B */
  int      dmy2[15]; /* to isolate cache line */
  int      deq;
  int      dmy3[15]; /* to isolate cache line */
  int      enq;
  int      dmy4[15]; /* to isolate cache line */
  float4D *slice;
  CNNet   *net;
  int      batch_size;
  int      nchan;
  int      insize;
} th_inference_args[MAX_NTHREAD];
volatile int th_inference_retv[MAX_NTHREAD];
pthread_t    th_inference_t[MAX_NTHREAD];
void         th_inference(struct th_inference_args *);

#endif  // CNNET_H_
